import numpy as np
import ikpy.chain
import os

URDF_PATH    = "/Users/parshg/Documents/lerobot/lerobot/examples/balance/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
OUTPUT_DIR   = "assets"
Q_NOMINAL_DEG = np.array([0.0, -100.0, 25.0, -90.0, 0.0])  # 5 joints (no gripper)

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def build_chain(urdf_path: str) -> ikpy.chain.Chain:
    """Load SO-101 URDF as a 5-joint revolute chain (gripper excluded)."""
    return ikpy.chain.Chain.from_urdf_file(
        urdf_path,
        active_links_mask=[False, True, True, True, True, True, False],
    )


def analytical_jacobian(chain: ikpy.chain.Chain, q_ikpy: np.ndarray) -> np.ndarray:
    """
    Compute the geometric Jacobian analytically using the revolute joint formula:

        J[:3, i] = zᵢ × (pₑₑ - pᵢ)   ← linear velocity
        J[3:, i] = zᵢ                  ← angular velocity

    where zᵢ is the joint rotation axis in world frame.
    All SO-101 joints rotate about local z-axis (from URDF <axis xyz="0 0 1">).
    """
    transforms = chain.forward_kinematics(q_ikpy, full_kinematics=True)
    p_ee = transforms[-1][:3, 3]

    active_indices = [i for i, l in enumerate(chain.links) if l.joint_type == "revolute"]
    J = np.zeros((6, len(active_indices)))

    for col, link_idx in enumerate(active_indices):
        R_i = transforms[link_idx][:3, :3]
        z_i = R_i @ np.array([0.0, 0.0, 1.0])   # local z → world frame
        p_i = transforms[link_idx][:3, 3]
        lever = p_ee - p_i
        J[:3, col] = np.cross(z_i, lever)        # linear velocity
        J[3:, col] = z_i                          # angular velocity

    return J


def main():
    # Load chain
    chain  = build_chain(URDF_PATH)
    q_rad  = np.deg2rad(Q_NOMINAL_DEG)
    q_ikpy = np.concatenate([[0], q_rad, [0]])

    # Compute J and J_pinv
    J      = analytical_jacobian(chain, q_ikpy)
    J_pinv = np.linalg.pinv(J)

    # Save
    j_path      = os.path.join(OUTPUT_DIR, "J_so101.npy")
    j_pinv_path = os.path.join(OUTPUT_DIR, "J_pinv_so101.npy")
    np.save(j_path,      J)
    np.save(j_pinv_path, J_pinv)

    np.set_printoptions(precision=4, suppress=True)
    print("J (6x5)  rows=[vx,vy,vz,ωx,ωy,ωz]  cols=[pan,lift,elbow,wrist_flex,wrist_roll]")
    print(J)
    print("\nJ_pinv (5x6)")
    print(J_pinv)
    print(f"\nSaved {j_path}")
    print(f"Saved {j_pinv_path}")


if __name__ == "__main__":
    main()