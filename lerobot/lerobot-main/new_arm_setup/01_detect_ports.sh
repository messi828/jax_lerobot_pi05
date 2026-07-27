#!/usr/bin/env bash
# 新一套 SO101 主从臂串口识别 + 生成 udev 规则
# 交互式：按提示逐个插入机械臂，脚本自动识别新出现的串口并读取序列号

set -u

list_ttys() { ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null | sort; }

get_serial() {
  udevadm info -q property -n "$1" | awk -F= '/^ID_SERIAL_SHORT=/{print $2}'
}
get_vid() {
  udevadm info -q property -n "$1" | awk -F= '/^ID_VENDOR_ID=/{print $2}'
}
get_pid() {
  udevadm info -q property -n "$1" | awk -F= '/^ID_MODEL_ID=/{print $2}'
}

wait_new_dev() {
  local before="$1"
  local newdev=""
  for _ in $(seq 1 30); do
    sleep 1
    newdev=$(comm -13 <(echo "$before") <(list_ttys) | head -1)
    [ -n "$newdev" ] && break
  done
  echo "$newdev"
}

echo "=== 第一步：请拔掉两条新机械臂的 USB 线，然后按回车 ==="
read -r
BASE=$(list_ttys)

echo "=== 第二步：请插入【从臂 follower】的 USB 线（等待识别中...） ==="
FOLLOWER_DEV=$(wait_new_dev "$BASE")
if [ -z "$FOLLOWER_DEV" ]; then echo "30 秒内未检测到新串口，退出"; exit 1; fi
F_SERIAL=$(get_serial "$FOLLOWER_DEV"); F_VID=$(get_vid "$FOLLOWER_DEV"); F_PID=$(get_pid "$FOLLOWER_DEV")
echo "  从臂: $FOLLOWER_DEV  vendor=$F_VID product=$F_PID serial=$F_SERIAL"

BASE=$(list_ttys)
echo "=== 第三步：请插入【主臂 leader】的 USB 线（等待识别中...） ==="
LEADER_DEV=$(wait_new_dev "$BASE")
if [ -z "$LEADER_DEV" ]; then echo "30 秒内未检测到新串口，退出"; exit 1; fi
L_SERIAL=$(get_serial "$LEADER_DEV"); L_VID=$(get_vid "$LEADER_DEV"); L_PID=$(get_pid "$LEADER_DEV")
echo "  主臂: $LEADER_DEV  vendor=$L_VID product=$L_PID serial=$L_SERIAL"

if [ "$F_SERIAL" = "$L_SERIAL" ]; then
  echo "警告：两个序列号相同，udev 规则无法区分，请检查！"
fi

RULES_FILE=/tmp/99-so101-serial-v2.rules
cat > "$RULES_FILE" <<EOF
# 新一套 SO101 主从臂（生成于 $(date +%F)）
SUBSYSTEM=="tty", ATTRS{idVendor}=="$F_VID", ATTRS{idProduct}=="$F_PID", ATTRS{serial}=="$F_SERIAL", SYMLINK+="so101_follower2", MODE="0666"
SUBSYSTEM=="tty", ATTRS{idVendor}=="$L_VID", ATTRS{idProduct}=="$L_PID", ATTRS{serial}=="$L_SERIAL", SYMLINK+="so101_leader2", MODE="0666"
EOF

echo
echo "udev 规则已生成到 $RULES_FILE，内容如下："
cat "$RULES_FILE"
echo
echo "请执行以下命令安装规则："
echo "  sudo cp $RULES_FILE /etc/udev/rules.d/99-so101-serial-v2.rules"
echo "  sudo udevadm control --reload-rules && sudo udevadm trigger"
echo
echo "之后新臂将固定为 /dev/so101_follower2 和 /dev/so101_leader2"
