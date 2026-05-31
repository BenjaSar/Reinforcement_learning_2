"""
Generate SUMO route files at configurable demand levels.
Maps the pure-Python arrival model (Poisson lambda=0.3 per lane per step)
to SUMO flow definitions on the 4×4 grid network boundary edges.

Usage:
    python sumo/generate_routes.py --demand 0.3 --output sumo/routes/grid4x4_03.rou.xml
"""
import argparse, os, xml.etree.ElementTree as ET


# For each boundary edge, define edge IDs through the grid to the opposite side.
# Edge naming: fromJunctionToJunction (e.g., A0B0 = from A0 to B0).
# Grid layout (coordinates in parens):
#   Row 0 (y=300): A0  B0  C0  D0
#   Row 1 (y=600): A1  B1  C1  D1
#   Row 2 (y=900): A2  B2  C2  D2
#   Row 3 (y=1200):A3  B3  C3  D3
ROUTES = {
    # From top → bottom (vertical, descending rows)
    "top1B3": ["top1B3", "B3B2", "B2B1", "B1B0", "B0bottom1"],
    "top2C3": ["top2C3", "C3C2", "C2C1", "C1C0", "C0bottom2"],
    "top3D3": ["top3D3", "D3D2", "D2D1", "D1D0", "D0bottom3"],
    # From bottom → top (vertical, ascending rows)
    "bottom1B0": ["bottom1B0", "B0B1", "B1B2", "B2B3", "B3top1"],
    "bottom2C0": ["bottom2C0", "C0C1", "C1C2", "C2C3", "C3top2"],
    "bottom3D0": ["bottom3D0", "D0D1", "D1D2", "D2D3", "D3top3"],
    # From left → right (horizontal, ascending columns)
    "left0A0": ["left0A0", "A0B0", "B0C0", "C0D0", "D0right0"],
    "left1A1": ["left1A1", "A1B1", "B1C1", "C1D1", "D1right1"],
    "left2A2": ["left2A2", "A2B2", "B2C2", "C2D2", "D2right2"],
    "left3A3": ["left3A3", "A3B3", "B3C3", "C3D3", "D3right3"],
    # From right → left (horizontal, descending columns)
    "right0D0": ["right0D0", "D0C0", "C0B0", "B0A0", "A0left0"],
    "right1D1": ["right1D1", "D1C1", "C1B1", "B1A1", "A1left1"],
    "right2D2": ["right2D2", "D2C2", "C2B2", "B2A2", "A2left2"],
    "right3D3": ["right3D3", "D3C3", "C3B3", "B3A3", "A3left3"],
    # Also add bottom0A0 → left (which was in the original file)
    "bottom0A0": ["bottom0A0", "A0left0"],
}

# Boundary edges of the 4×4 grid from sumo-rl's grid4x4 network
BOUNDARY_EDGES = sorted(ROUTES.keys())

# Base arrival rate matching Python env: 0.3 veh/step/lane, step=5s
# vehsPerHour = 0.3 * demand * (3600 / 5) * lanes_per_boundary
# Each boundary serves one intersection (4 lanes), but some share lanes
# Scale: at demand=1.0, total arrival ≈ 64 lanes × 0.3 × 720/step ÷ 5s = ~2765 veh/hour
# Distribute across N boundaries
VEHS_PER_HOUR_AT_DEMAND_1 = 300  # total scaled for the network


def generate_route_file(demand: float, output_path: str, num_seconds: int = 3600):
    """Generate a .rou.xml file with flows scaled to demand level."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    root = ET.Element("routes")
    root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    root.set("xsi:noNamespaceSchemaLocation",
             "http://sumo.dlr.de/xsd/routes_file.xsd")

    n_boundaries = len(BOUNDARY_EDGES)
    flow_rate = (VEHS_PER_HOUR_AT_DEMAND_1 * demand) / n_boundaries

    for edge in BOUNDARY_EDGES:
        route_edges = ROUTES.get(edge, [edge])
        route_elem = ET.SubElement(root, "route")
        route_elem.set("id", f"route_{edge}")
        route_elem.set("edges", " ".join(route_edges))

        flow_elem = ET.SubElement(root, "flow")
        flow_elem.set("id", f"flow_{edge}")
        flow_elem.set("route", f"route_{edge}")
        flow_elem.set("begin", "0")
        flow_elem.set("end", str(num_seconds))
        flow_elem.set("vehsPerHour", f"{flow_rate:.1f}")

    tree = ET.ElementTree(root)
    tree.write(output_path, encoding="UTF-8", xml_declaration=True)
    print(f"[Routes] Generated {output_path} (demand={demand}, "
          f"{n_boundaries} flows @ {flow_rate:.1f} veh/hr each)")


def generate_sumocfg(demand: float, output_path: str, net_path: str,
                     route_path: str, num_seconds: int = 3600):
    """Generate a .sumocfg file for the given route."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    root = ET.Element("configuration")
    input_el = ET.SubElement(root, "input")
    net_el = ET.SubElement(input_el, "net-file")
    net_el.set("value", net_path)
    route_el = ET.SubElement(input_el, "route-files")
    route_el.set("value", route_path)
    time_el = ET.SubElement(root, "time")
    begin_el = ET.SubElement(time_el, "begin")
    begin_el.set("value", "0")
    end_el = ET.SubElement(time_el, "end")
    end_el.set("value", str(num_seconds))

    tree = ET.ElementTree(root)
    tree.write(output_path, encoding="UTF-8", xml_declaration=True)
    print(f"[Config] Generated {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate SUMO routes")
    parser.add_argument("--demand", type=float, default=0.3,
                        help="Demand factor (0.0–1.0)")
    parser.add_argument("--num_seconds", type=int, default=3600)
    parser.add_argument("--output_dir", type=str, default="sumo/routes")
    parser.add_argument("--net_file", type=str,
                        default="sumo/grid4x4.net.xml")
    args = parser.parse_args()

    demand_str = f"{args.demand:.1f}".replace(".", "")
    route_path = os.path.join(args.output_dir,
                              f"grid4x4_{demand_str}.rou.xml")
    cfg_path = os.path.join(args.output_dir,
                            f"grid4x4_{demand_str}.sumocfg")

    generate_route_file(args.demand, route_path, args.num_seconds)
    generate_sumocfg(args.demand, cfg_path, args.net_file,
                     os.path.basename(route_path), args.num_seconds)


if __name__ == "__main__":
    main()
