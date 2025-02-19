import pandas as pd

def print_state_jacobian(jacobian_state, model):

    nq = model.nq  # expected to be 7
    nv = model.nv  # expected to be 6
    # Define labels for qpos and qvel
    qpos_labels = ['p_x', 'p_y', 'p_z', 'q_w', 'q_x', 'q_y', 'q_z']
    qvel_labels = ['v_x', 'v_y', 'v_z', 'ω_x', 'ω_y', 'ω_z']

    # Extract blocks from the full Jacobian
    # dq_next/dq: top-left block (nq x nq)
    dq_dq = jacobian_state[:nq, :nq]
    # dq_next/dv: top-right block (nq x nv)
    dq_dv = jacobian_state[:nq, nq:]
    # dv_next/dq: bottom-left block (nv x nq)
    dv_dq = jacobian_state[nq:, :nq]
    # dv_next/dv: bottom-right block (nv x nv)
    dv_dv = jacobian_state[nq:, nq:]

    # Create DataFrames for better formatting in terminal output
    df_dq_dq = pd.DataFrame(dq_dq, index=qpos_labels, columns=qpos_labels)
    df_dq_dv = pd.DataFrame(dq_dv, index=qpos_labels, columns=qvel_labels)
    df_dv_dq = pd.DataFrame(dv_dq, index=qvel_labels, columns=qpos_labels)
    df_dv_dv = pd.DataFrame(dv_dv, index=qvel_labels, columns=qvel_labels)

    # Print the blocks with headers
    print("Jacobian Block: dq_next/dq (Position w.r.t. Position)")
    print(df_dq_dq)
    print("\nJacobian Block: dq_next/dv (Position w.r.t. Velocity)")
    print(df_dq_dv)
    print("\nJacobian Block: dv_next/dq (Velocity w.r.t. Position)")
    print(df_dv_dq)
    print("\nJacobian Block: dv_next/dv (Velocity w.r.t. Velocity)")
    print(df_dv_dv)
