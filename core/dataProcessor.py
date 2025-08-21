import re
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class TouchData:
    """触摸数据类"""

    frame_number: int
    timestamp: Optional[str] = None
    tx_count: int = 0
    rx_count: int = 0
    data_matrix: Optional[np.ndarray] = None

    def __str__(self):
        return f"Frame {self.frame_number}: TX={self.tx_count}, RX={self.rx_count}"


@dataclass
class HeaderInfo:
    """头部信息数据类"""

    date: str
    time: str
    fw_version: str
    tx: int
    rx: int
    tx_voltage: int
    cf: int

    def __str__(self):
        return f"Header: {self.date} {self.time}, TX={self.tx}, RX={self.rx}"


class TouchDataParser:
    """触摸数据解析器"""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.header_info: Optional[HeaderInfo] = None
        self.touch_data: List[TouchData] = []

    def parse(self) -> Tuple[Optional[HeaderInfo], List[TouchData]]:
        """解析文件中的触摸数据"""
        try:
            with open(self.file_path, "r", encoding="utf-8") as file:
                content = file.read()
                self._parse_content(content)
        except FileNotFoundError:
            print(f"文件未找到: {self.file_path}")
        except Exception as e:
            print(f"解析文件时出错: {e}")

        return self.header_info, self.touch_data

    def _parse_content(self, content: str):
        """解析文件内容"""
        lines = content.split("\n")

        # 解析头部信息
        self._parse_header(lines)

        # 解析触摸数据
        self._parse_touch_data(lines)

    def _parse_header(self, lines: List[str]):
        """解析头部信息"""
        header_data = {}

        for line in lines:
            line = line.strip()
            if line.startswith("DATE,"):
                header_data["date"] = line.split(",")[1].strip()
            elif line.startswith("TIME,"):
                header_data["time"] = line.split(",")[1].strip()
            elif line.startswith("FW Ver:"):
                header_data["fw_version"] = line.split(":", 1)[1].strip()
            elif line.startswith("TX,"):
                header_data["tx"] = int(line.split(",")[1].strip())
            elif line.startswith("RX,"):
                header_data["rx"] = int(line.split(",")[1].strip())
            elif line.startswith("TX_VOLTAGE,"):
                header_data["tx_voltage"] = int(line.split(",")[1].strip())
            elif line.startswith("Cf,"):
                header_data["cf"] = int(line.split(",")[1].strip())

        if len(header_data) >= 4:  # 至少需要基本信息
            self.header_info = HeaderInfo(
                date=header_data.get("date", ""),
                time=header_data.get("time", ""),
                fw_version=header_data.get("fw_version", ""),
                tx=header_data.get("tx", 0),
                rx=header_data.get("rx", 0),
                tx_voltage=header_data.get("tx_voltage", 0),
                cf=header_data.get("cf", 0),
            )

    def _parse_touch_data(self, lines: List[str]):
        """解析触摸数据"""
        current_frame = None
        current_data = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 检查是否是新的帧开始
            if line.startswith("MC Diff: Frame"):
                # 保存上一帧数据
                if current_frame is not None and current_data:
                    self._save_frame_data(current_frame, current_data)

                # 开始新帧
                frame_match = re.search(r"Frame (\d+)", line)
                if frame_match:
                    current_frame = int(frame_match.group(1))
                    current_data = []

            # 解析数据行
            elif line.startswith("MC_TX"):
                data_row = self._parse_data_row(line)
                if data_row is not None:
                    current_data.append(data_row)

        # 保存最后一帧数据
        if current_frame is not None and current_data:
            self._save_frame_data(current_frame, current_data)

    def _parse_data_row(self, line: str) -> Optional[List[int]]:
        """解析数据行"""
        try:
            # 移除TX标识符，获取数据部分
            data_part = re.sub(r"MC_TX\d+,", "", line)

            # 分割数据并转换为整数
            values = []
            for value in data_part.split(","):
                value = value.strip()
                if value:
                    values.append(int(value))

            return values if values else None
        except (ValueError, IndexError):
            return None

    def _save_frame_data(self, frame_number: int, data_rows: List[List[int]]):
        """保存帧数据"""
        if not data_rows:
            return

        # 转换为numpy数组
        try:
            data_matrix = np.array(data_rows)
            tx_count, rx_count = data_matrix.shape

            touch_data = TouchData(
                frame_number=frame_number,
                tx_count=tx_count,
                rx_count=rx_count,
                data_matrix=data_matrix,
            )

            self.touch_data.append(touch_data)
        except Exception as e:
            print(f"保存帧数据时出错: {e}")

    def get_frame_by_number(self, frame_number: int) -> Optional[TouchData]:
        """根据帧号获取数据"""
        for frame in self.touch_data:
            if frame.frame_number == frame_number:
                return frame
        return None

    def get_data_statistics(self) -> Dict:
        """获取数据统计信息"""
        if not self.touch_data:
            return {}

        all_values = []
        for frame in self.touch_data:
            if frame.data_matrix is not None:
                all_values.extend(frame.data_matrix.flatten())

        if not all_values:
            return {}

        all_values = np.array(all_values)

        return {
            "total_frames": len(self.touch_data),
            "tx_count": self.touch_data[0].tx_count if self.touch_data else 0,
            "rx_count": self.touch_data[0].rx_count if self.touch_data else 0,
            "data_range": f"{all_values.min()} - {all_values.max()}",
            "data_mean": float(all_values.mean()),
            "data_std": float(all_values.std()),
            "positive_values": int(np.sum(all_values > 0)),
            "negative_values": int(np.sum(all_values < 0)),
            "zero_values": int(np.sum(all_values == 0)),
        }

    def get_frame_data_as_array(self, frame_number: int) -> Optional[np.ndarray]:
        """获取指定帧的数据数组"""
        frame = self.get_frame_by_number(frame_number)
        return frame.data_matrix if frame else None

    def normalize_data(self, method: str = "minmax") -> List[TouchData]:
        """数据归一化"""
        normalized_data = []

        for frame in self.touch_data:
            if frame.data_matrix is None:
                continue

            if method == "minmax":
                # Min-Max归一化
                data_min = frame.data_matrix.min()
                data_max = frame.data_matrix.max()
                if data_max != data_min:
                    normalized_matrix = (frame.data_matrix - data_min) / (
                        data_max - data_min
                    )
                else:
                    normalized_matrix = frame.data_matrix
            elif method == "zscore":
                # Z-score归一化
                mean = frame.data_matrix.mean()
                std = frame.data_matrix.std()
                if std != 0:
                    normalized_matrix = (frame.data_matrix - mean) / std
                else:
                    normalized_matrix = frame.data_matrix
            else:
                normalized_matrix = frame.data_matrix

            normalized_frame = TouchData(
                frame_number=frame.frame_number,
                timestamp=frame.timestamp,
                tx_count=frame.tx_count,
                rx_count=frame.rx_count,
                data_matrix=normalized_matrix,
            )
            normalized_data.append(normalized_frame)

        return normalized_data

    def save_to_file(self, file_path):
        """保存触摸数据到文件"""
        try:
            with open(file_path, "w", encoding="utf-8") as file:
                # 写入头部信息
                if self.header_info:
                    file.write(f"Date: {self.header_info.date}\n")
                    file.write(f"Time: {self.header_info.time}\n")
                    file.write(f"FW Version: {self.header_info.fw_version}\n")
                    file.write(f"TX: {self.header_info.tx}\n")
                    file.write(f"RX: {self.header_info.rx}\n")
                    file.write(f"TX voltage: {self.header_info.tx_voltage}\n")
                    file.write(f"CF: {self.header_info.cf}\n\n")

                # 写入每一帧数据
                for frame in self.touch_data:
                    if frame.data_matrix is not None:
                        file.write(f"MC Diff: Frame {frame.frame_number}\n")

                        # 写入数据矩阵
                        for tx_idx in range(frame.tx_count):
                            row_values = ",".join(map(str, frame.data_matrix[tx_idx]))
                            file.write(f"\tMC_TX{tx_idx:02d}, {row_values}\n")

                        file.write("\n")

            return True
        except Exception as e:
            print(f"保存文件时出错: {e}")
            return False


# 使用示例
if __name__ == "__main__":
    # 创建解析器实例
    parser = TouchDataParser("./data/亮屏1指.txt")

    # 解析文件
    header_info, touch_data = parser.parse()

    # 打印头部信息
    if header_info:
        print("头部信息:")
        print(f"  {header_info}")

    # 打印数据统计信息
    stats = parser.get_data_statistics()
    print("\n数据统计信息:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # 打印前几帧信息
    print(f"\n前{min(3, len(touch_data))}帧信息:")
    for frame in touch_data[:3]:
        print(f"  {frame}")
        if frame.data_matrix is not None:
            print(f"    数据形状: {frame.data_matrix.shape}")
            print(
                f"    数据范围: {frame.data_matrix.min()} - {frame.data_matrix.max()}"
            )

    # 获取特定帧的数据
    if touch_data:
        frame_0_data = parser.get_frame_data_as_array(0)
        if frame_0_data is not None:
            print(f"\n第0帧数据形状: {frame_0_data.shape}")
            print(f"第0帧数据前5行前5列:")
            print(frame_0_data[:5, :5])

    # 保存数据到文件示例
    parser.save_to_file("./data/parsed_touch_data.txt")
