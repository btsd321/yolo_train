import argparse
from pathlib import Path
import xml.etree.ElementTree as ET


def parse_points(points_str: str):
	cleaned = points_str.strip().rstrip(";")
	if not cleaned:
		return []
	coords = []
	for chunk in cleaned.split(";"):
		chunk = chunk.strip()
		if not chunk:
			continue
		chunk = chunk.strip("()")
		try:
			x_str, y_str = chunk.split(",")
			coords.append((float(x_str), float(y_str)))
		except ValueError:
			continue
	return coords


def area_to_bbox(area_elem: ET.Element):
	coords = parse_points(area_elem.get("points", ""))
	if not coords:
		return None
	xs = [c[0] for c in coords]
	ys = [c[1] for c in coords]
	return {
		"x": min(xs),
		"y": min(ys),
		"w": max(xs) - min(xs),
		"h": max(ys) - min(ys),
	}


def indent(elem: ET.Element, level: int = 0):
	# ET.indent is not available in older Python versions used in some environments
	i = "\n" + level * "    "
	if len(elem):
		if not elem.text or not elem.text.strip():
			elem.text = i + "    "
		for child in elem:
			indent(child, level + 1)
		if not elem.tail or not elem.tail.strip():
			elem.tail = i
	else:
		if level and (not elem.tail or not elem.tail.strip()):
			elem.tail = i


def convert_file(src_path: Path, dst_path: Path):
	tree = ET.parse(src_path)
	root = tree.getroot()

	rect_indices = []
	for rect in root.findall("rect"):
		try:
			rect_indices.append(int(rect.get("index", "0")))
		except ValueError:
			continue
	next_index = max(rect_indices) if rect_indices else 0

	areas = list(root.findall("area"))
	for area in areas:
		bbox = area_to_bbox(area)
		if not bbox:
			root.remove(area)
			continue

		next_index += 1
		rect_attrs = {
			"index": str(next_index),
			"x": str(bbox["x"]),
			"y": str(bbox["y"]),
			"h": str(bbox["h"]),
			"w": str(bbox["w"]),
			"text": area.get("text", ""),
			"labelType": area.get("labelType", ""),
		}
		rect_elem = ET.Element("rect", rect_attrs)
		root.append(rect_elem)
		root.remove(area)

	indent(root)
	tree.write(dst_path, encoding="UTF-8", xml_declaration=True)


def run(input_dir: Path, output_dir: Path):
	xml_files = sorted(p for p in input_dir.glob("*.xml") if p.is_file())
	if not xml_files:
		print(f"No xml files found in {input_dir}")
		return

	output_dir.mkdir(parents=True, exist_ok=True)

	for xml_path in xml_files:
		dst_path = output_dir / xml_path.name
		convert_file(xml_path, dst_path)
		print(f"Converted {xml_path.name} -> {dst_path}")


def parse_args():
	parser = argparse.ArgumentParser(
		description="Convert area polygons in XML annotations to rect bounding boxes"
	)
	parser.add_argument("--input", required=True, type=Path, help="Input folder with XML files")
	parser.add_argument(
		"--output",
		type=Path,
		default=None,
		help="Output folder for converted XML files (default: input/bbox)",
	)
	return parser.parse_args()


def main():
	args = parse_args()
	input_dir: Path = args.input
	if not input_dir.is_dir():
		raise SystemExit(f"Input path {input_dir} is not a directory")

	output_dir = args.output or (input_dir / "bbox")
	run(input_dir, output_dir)


if __name__ == "__main__":
	main()
