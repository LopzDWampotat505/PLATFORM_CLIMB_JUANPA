"""Convert ANYbotics' flat-text ANYdrive 3.6 actuator net (anydrive_3_6_20210303.txt)
into a TorchScript .pt model consumable by isaaclab.actuators.ActuatorNetMLP.

Architecture (recovered from the C++ source, SimpleMLPLinearOutput / ActuationModel,
in anymal_ctrl_rl_isaac_lab/include/anymal_ctrl_rl_isaac_lab/actuator_model/):
    input (6) -> Linear(6,32) -> Softsign -> Linear(32,32) -> Softsign -> Linear(32,1)

File layout (1313 floats, single comma-separated line):
    W0 (32x6, row-major) , b0 (32) , W1 (32x32, row-major) , b1 (32) , W2 (1x32, row-major) , b2 (1)

Usage:
    python3 convert_actuator_net.py <anydrive_3_6_20210303.txt> <anydrive_3_6_mlp_jit.pt>

Run inside the Isaac Lab conda env (needs torch).
"""

import sys
import torch
import torch.nn as nn

INPUT_TXT = sys.argv[1] if len(sys.argv) > 1 else "anydrive_3_6_20210303.txt"
OUTPUT_PT = sys.argv[2] if len(sys.argv) > 2 else "anydrive_3_6_mlp_jit.pt"

IN_DIM = 6
HIDDEN = 32
OUT_DIM = 1


class AnydriveMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(IN_DIM, HIDDEN)
        self.act1 = nn.Softsign()
        self.l2 = nn.Linear(HIDDEN, HIDDEN)
        self.act2 = nn.Softsign()
        self.l3 = nn.Linear(HIDDEN, OUT_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.l1(x))
        x = self.act2(self.l2(x))
        return self.l3(x)


def main():
    with open(INPUT_TXT) as f:
        vals = [float(v) for v in f.read().strip().split(",")]

    expected = IN_DIM * HIDDEN + HIDDEN + HIDDEN * HIDDEN + HIDDEN + HIDDEN * OUT_DIM + OUT_DIM
    assert len(vals) == expected, f"got {len(vals)} floats, expected {expected}"

    t = torch.tensor(vals, dtype=torch.float32)
    idx = 0

    def take(n):
        nonlocal idx
        chunk = t[idx : idx + n]
        idx += n
        return chunk

    w0 = take(HIDDEN * IN_DIM).view(HIDDEN, IN_DIM)
    b0 = take(HIDDEN)
    w1 = take(HIDDEN * HIDDEN).view(HIDDEN, HIDDEN)
    b1 = take(HIDDEN)
    w2 = take(OUT_DIM * HIDDEN).view(OUT_DIM, HIDDEN)
    b2 = take(OUT_DIM)

    model = AnydriveMLP()
    with torch.no_grad():
        model.l1.weight.copy_(w0)
        model.l1.bias.copy_(b0)
        model.l2.weight.copy_(w1)
        model.l2.bias.copy_(b1)
        model.l3.weight.copy_(w2)
        model.l3.bias.copy_(b2)
    model.eval()

    scripted = torch.jit.script(model)
    torch.jit.save(scripted, OUTPUT_PT)

    # sanity check: zero input -> should be a small-ish bias-only torque
    with torch.no_grad():
        out = scripted(torch.zeros(1, IN_DIM))
    print(f"Saved {OUTPUT_PT}. Sanity check output at zero input: {out.item():.4f}")


if __name__ == "__main__":
    main()
