#!/usr/bin/env python3
"""ZMQ-based policy server for ARX arm deployment.

Usage:
    python scripts/serve_policy_zmq.py \
        --config-name pi05_arx \
        --checkpoint-dir /path/to/checkpoint \
        --port 6789
"""

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import zmq

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from openpi.training import config as _config
from openpi.policies import policy_config

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ARXPolicyServer:
    """ZMQ server for ARX policy inference."""
    
    def __init__(self, config_name: str, checkpoint_dir: str, host: str = "localhost", port: int = 6789):
        """Initialize policy server.
        
        Args:
            config_name: Training config name (e.g., 'pi05_arx')
            checkpoint_dir: Path to checkpoint directory
            host: Host to bind (default: 'localhost' for local connections)
            port: ZMQ port to bind
        """
        self.host = host
        self.port = port
        
        # Load policy
        print(f"Loading policy from config: {config_name}")
        print(f"Checkpoint directory: {checkpoint_dir}")
        
        config = _config.get_config(config_name)
        self.policy = policy_config.create_trained_policy(config, checkpoint_dir)
        
        print("Policy loaded successfully")
        
        # Initialize ZMQ
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.bind(f"tcp://{host}:{port}")
        print(f"ZMQ server started on {host}:{port}")
    
    def process_observation(self, obs: dict) -> dict:
        """Process observation from client to policy input format.
        
        This converts your robot's observation format to the format expected by policy.infer().
        
        Expected client obs format:
        {
            'rgb': {'front': np.ndarray (H, W, 3), 'wrist': np.ndarray (H, W, 3)},
            'state': np.ndarray (7,),  # [x, y, z, roll, pitch, yaw, gripper]
            'instruct': str
        }
        
        Returns policy input format (matches training data after repack_transforms):
        {
            'observation.images.front': np.ndarray,
            'observation.images.wrist': np.ndarray,
            'observation.state': np.ndarray (27,),
            'prompt': str
        }
        
        Data flow:
        1. Your robot state (7D) → observation.state (27D)
        2. policy.infer() receives this dict
        3. ARXInputs transform: 27D → extract 7D → pad to 32D
        4. Normalize: apply norm_stats
        5. Model inference
        6. ARXOutputs transform: 32D → extract 7D
        7. Unnormalize: reverse norm_stats
        8. Return actions (7D)
        """
        # Extract RGB images
        front_image = obs['rgb']['front']  # (H, W, 3) BGR uint8
        wrist_image = obs['rgb']['wrist']  # (H, W, 3) BGR uint8, optional
        
        available_cameras = list(obs['rgb'].keys())
        if wrist_image is not None:
            logger.debug(f"Received images from cameras: {available_cameras}")
        else:
            logger.debug(f"Received images from cameras: {available_cameras} (wrist not available)")
        
        # Extract state (7D: ee_pose + gripper)
        state_7d = np.array(obs['state'], dtype=np.float32)
        
        # IMPORTANT: Ensure your state values match training data units and ranges
        # - Position: mm or m? (check your training data)
        # - Angles: degrees or radians? (check your training data)
        # - Gripper: raw value or normalized? (check your training data)
        
        # Pad state to 27D (match dataset format)
        # Dataset structure (verified from your data):
        # - Indices 0-5:   end_effector pose (x, y, z, roll, pitch, yaw)
        # - Indices 6-23:  6 joints × (pos, vel, cur) - we don't have this, pad with zeros
        # - Indices 24-26: gripper × (pos, vel, cur) - we only have pos
        state_27d = np.zeros(27, dtype=np.float32)
        state_27d[0:6] = state_7d[0:6]  # end_effector pose
        state_27d[24] = state_7d[6]     # gripper position
        # Indices 6-23 and 25-26 remain zero (joint velocities/currents, gripper vel/cur)
        
        # Extract instruction
        instruction = obs.get('instruct', '')
        
        # Build policy input
        # This format matches what ARXInputs expects (after repack_transforms)
        policy_input = {
            'observation.images.front': front_image,
            'observation.state': state_27d,
            'prompt': instruction,
        }
        
        # Add wrist image if available
        if wrist_image is not None:
            policy_input['observation.images.wrist'] = wrist_image
        
        return policy_input
    
    def run(self):
        """Main server loop."""
        print("Server ready, waiting for clients...")
        
        try:
            while True:
                # Receive message
                message = self.socket.recv_multipart()
                client_id = message[0]
                b_obs = message[1]
                
                # Handle handshake
                if b_obs == b'Agx Ready to Start Work':
                    self.socket.send_multipart([client_id, b'Model ready'])
                    continue
                
                # Deserialize observation
                try:
                    obs = pickle.loads(b_obs)
                except Exception as e:
                    logger.error(f"Failed to deserialize observation: {e}")
                    continue
                
                # Process observation
                try:
                    policy_input = self.process_observation(obs)
                except Exception as e:
                    logger.error(f"Failed to process observation: {e}")
                    # Send zero action on error
                    response = {
                        'action': np.zeros(7, dtype=np.float32).tolist(),
                        'done': False
                    }
                    self.socket.send_multipart([client_id, pickle.dumps(response)])
                    continue
                
                # Run inference
                start_time = time.time()
                try:
                    result = self.policy.infer(policy_input)
                    actions = result['actions']  # (action_horizon, 7) or (7,)
                    raw_actions = result['raw_actions']
                    # Ensure actions is always 2D: (action_horizon, 7)
                    if actions.ndim == 1:
                        actions = actions.reshape(1, -1)  # (7,) -> (1, 7)
                    if raw_actions.ndim == 1:
                        raw_actions = raw_actions.reshape(1, -1)  # (7,) -> (1, 7)
                    
                    print(f"Actions shape: {actions.shape}")
                    inference_time = time.time() - start_time
                    
                except Exception as e:
                    logger.error(f"Inference failed: {e}")
                    import traceback
                    traceback.print_exc()
                    # Send zero action on error (1, 7)
                    actions = np.zeros((1, 7), dtype=np.float32)
                
                # Send response
                response = {
                    'actions': actions.tolist(),  # 2D list: [[a1, a2, ..., a7], ...]
                    'raw_actions': raw_actions.tolist(),
                    'done': False
                }
                b_response = pickle.dumps(response)
                self.socket.send_multipart([client_id, b_response])
                
        except KeyboardInterrupt:
            print("\nServer interrupted by user")
        finally:
            self.socket.close()
            self.context.term()
            print("Server shutdown complete")


def main():
    parser = argparse.ArgumentParser(description="ZMQ-based policy server for ARX arm")
    parser.add_argument(
        "--config-name",
        type=str,
        required=True,
        help="Training config name (e.g., 'pi05_arx')"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Path to checkpoint directory"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to bind (default: 'localhost', use '*' for all interfaces)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6789,
        help="ZMQ port to bind (default: 6789)"
    )
    
    args = parser.parse_args()
    
    # Create and run server
    server = ARXPolicyServer(
        config_name=args.config_name,
        checkpoint_dir=args.checkpoint_dir,
        host=args.host,
        port=args.port
    )
    server.run()


if __name__ == "__main__":
    main()
