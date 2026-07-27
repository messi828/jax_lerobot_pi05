# Leader Direct Record

This folder contains a standalone recording path for **SO101 leader -> SO101 follower**
that does **not** depend on `gym_manipulator` teleop event interfaces.

## Files

- `record_with_leader.py`: main script
- `leader_record_config.json`: config template

## Run

From repo root:

```bash
cd /home/jaylen/桌面/lerobot_rl/lerobot-main
python leader/record_with_leader.py --config_path leader/leader_record_config.json
```

## Episode labels

At the end of each episode:

- `s`: success
- `f`: failure
- `r`: rerecord current episode (discard buffer)
- `q`: quit (discard current buffer and exit)

## Notes

- Leader action is read from SO101 leader and directly sent to follower.
- Cameras are recorded from follower observations.
- Final reward/done label is written in one terminal frame per episode.

