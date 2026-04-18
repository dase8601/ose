"""
explore_arcagi.py — Explore ARC-AGI-3 environment for JEPA compatibility.

Based on the official ARC-AGI-3 agent API (arcprize/ARC-AGI-3-Agents):
- FrameData.frame = list of 2D numpy arrays (grids with color values 0-10)
- GameAction: RESET, ACTION1-4 (directional), ACTION5 (interact), ACTION6 (click)
- GameState: NOT_PLAYED, NOT_FINISHED, GAME_OVER, WIN

Tests:
1. What does the frame/grid data look like? (shape, type, unique values)
2. What actions are available?
3. Does DINOv2 produce discriminative features for different game states?
4. Can our world model architecture work here?

Usage:
    pip install arc-agi    # requires Python >= 3.12
    python explore_arcagi.py
"""

import numpy as np
from PIL import Image

# ARC-AGI color palette (matches arc_agi.CSS_PALETTE / ANSI_PALETTE)
ARC_COLORS = {
    0: [0, 0, 0],         # black
    1: [0, 116, 217],     # blue
    2: [255, 65, 54],     # red
    3: [46, 204, 64],     # green
    4: [255, 220, 0],     # yellow
    5: [170, 170, 170],   # gray
    6: [240, 18, 190],    # magenta
    7: [255, 133, 27],    # orange
    8: [127, 219, 255],   # light blue / cyan
    9: [135, 12, 37],     # maroon
    10: [255, 255, 255],  # white
}


def grid_to_rgb(grid, cell_size=16):
    """Convert a 2D grid of color indices to an RGB image.

    Each cell is rendered as cell_size x cell_size pixels.
    """
    grid = np.array(grid)
    h, w = grid.shape
    img = np.zeros((h * cell_size, w * cell_size, 3), dtype=np.uint8)
    for val, color in ARC_COLORS.items():
        mask = grid == val
        for dy in range(cell_size):
            for dx in range(cell_size):
                img[np.where(mask)[0] * cell_size + dy,
                    np.where(mask)[1] * cell_size + dx] = color
    return img


def grid_to_rgb_fast(grid, target_size=224):
    """Convert a 2D grid to an RGB image, resized for DINOv2."""
    grid = np.array(grid)
    h, w = grid.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for val, color in ARC_COLORS.items():
        img[grid == val] = color
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((target_size, target_size), Image.NEAREST)
    return np.array(pil_img)


print("=" * 60)
print("ARC-AGI-3 Exploration — Testing JEPA compatibility")
print("=" * 60)

# ── Step 1: Load the environment ─────────────────────────────────────────────
print("\n[1] Loading ARC-AGI-3 environment...")

try:
    import arc_agi
    from arcengine import GameAction, GameState
    try:
        from arcengine import FrameData, FrameDataRaw
        print("  FrameData/FrameDataRaw imported")
    except ImportError:
        FrameData = None
        FrameDataRaw = None
        print("  FrameData not available as separate import")
except ImportError as e:
    print(f"  FAILED: {e}")
    print("  Run: pip install arc-agi")
    print("  Note: arcengine requires Python >= 3.12")
    exit(1)

arc = arc_agi.Arcade()
env = arc.make("ls20", render_mode=None)  # None = no terminal rendering, faster

print("  Environment created: ls20 (render_mode=None for speed)")

# ── Step 2: Explore the environment API ──────────────────────────────────────
print("\n[2] Environment API exploration...")

env_attrs = [a for a in dir(env) if not a.startswith("_")]
print(f"  env attributes: {env_attrs}")

# GameAction members
print(f"  GameAction members: {[a.name for a in GameAction]}")

# GameState members
print(f"  GameState members: {[s.name for s in GameState]}")

# ── Step 3: Reset and inspect frame data ────────────────────────────────────
print("\n[3] Resetting environment and inspecting frame data...")

# The ARC-AGI-3 API: env.step(action) returns FrameDataRaw or similar
# with .frame (list of 2D arrays), .state, .levels_completed

# First reset
try:
    result = env.step(GameAction.RESET)
    print(f"  RESET returned: type={type(result).__name__}")

    # Inspect the result object
    if hasattr(result, 'frame'):
        frames = result.frame
        print(f"  result.frame: type={type(frames).__name__}, len={len(frames) if hasattr(frames, '__len__') else 'N/A'}")
        for i, f in enumerate(frames if isinstance(frames, list) else [frames]):
            f_arr = np.array(f)
            print(f"    frame[{i}]: shape={f_arr.shape}, dtype={f_arr.dtype}, "
                  f"min={f_arr.min()}, max={f_arr.max()}, unique={np.unique(f_arr).tolist()[:15]}")

    if hasattr(result, 'state'):
        print(f"  result.state: {result.state}")
    if hasattr(result, 'levels_completed'):
        print(f"  result.levels_completed: {result.levels_completed}")
    if hasattr(result, 'available_actions'):
        print(f"  result.available_actions: {result.available_actions}")
    if hasattr(result, 'game_id'):
        print(f"  result.game_id: {result.game_id}")

    # If result is a dict-like or has other structure
    if isinstance(result, dict):
        print(f"  dict keys: {list(result.keys())}")
        for k, v in result.items():
            if isinstance(v, (list, np.ndarray)):
                v_arr = np.array(v) if isinstance(v, list) else v
                if v_arr.ndim >= 2:
                    print(f"    {k}: shape={v_arr.shape}")
            else:
                print(f"    {k}: {type(v).__name__} = {str(v)[:100]}")

    # Try all attributes
    print("\n  All result attributes with values:")
    for attr in dir(result):
        if attr.startswith("_"):
            continue
        try:
            val = getattr(result, attr)
            if callable(val):
                continue
            print(f"    .{attr} = {str(val)[:150]}")
        except Exception:
            pass

except Exception as e:
    print(f"  RESET failed: {e}")
    import traceback
    traceback.print_exc()

# ── Step 4: Collect diverse game states ─────────────────────────────────────
print("\n[4] Collecting diverse game states across multiple actions...")

grids = []  # Will store 2D grid arrays
states_info = []  # Will store metadata about each state

# Take varied actions to get diverse states
action_sequence = [
    GameAction.ACTION1,  # up
    GameAction.ACTION2,  # down
    GameAction.ACTION3,  # left
    GameAction.ACTION4,  # right
    GameAction.ACTION1,
    GameAction.ACTION3,
    GameAction.ACTION2,
    GameAction.ACTION4,
    GameAction.ACTION1,
    GameAction.ACTION1,
]

for i, action in enumerate(action_sequence):
    try:
        result = env.step(action)

        # Extract grid data from result
        grid = None
        state_str = "unknown"

        if hasattr(result, 'frame') and result.frame is not None:
            frames = result.frame
            if isinstance(frames, list) and len(frames) > 0:
                grid = np.array(frames[0])
            elif hasattr(frames, 'shape'):
                grid = np.array(frames)
        elif isinstance(result, dict) and 'frame' in result:
            frames = result['frame']
            if isinstance(frames, list) and len(frames) > 0:
                grid = np.array(frames[0])

        if hasattr(result, 'state'):
            state_str = str(result.state)
        elif isinstance(result, dict) and 'state' in result:
            state_str = str(result['state'])

        if grid is not None and grid.ndim == 2:
            grids.append(grid)
            states_info.append({
                'action': action.name if hasattr(action, 'name') else str(action),
                'state': state_str,
                'shape': grid.shape,
                'unique_vals': np.unique(grid).tolist(),
            })
            print(f"  Step {i} ({action.name if hasattr(action, 'name') else action}): "
                  f"grid={grid.shape}, state={state_str}, "
                  f"unique={np.unique(grid).tolist()[:10]}")
        else:
            # Try to extract from any available attribute
            print(f"  Step {i}: grid=None, result_type={type(result).__name__}")
            if grid is not None:
                print(f"    grid ndim={grid.ndim}, shape={grid.shape}")

    except Exception as e:
        print(f"  Step {i} failed: {e}")

print(f"\n  Collected {len(grids)} grid states")

# ── Step 5: Visualize grids (save as images) ──────────────────────────────
print("\n[5] Rendering grids to RGB images...")

rgb_images = []
for i, grid in enumerate(grids):
    rgb = grid_to_rgb_fast(grid, target_size=224)
    rgb_images.append(rgb)
    print(f"  Grid {i}: {grid.shape} → RGB {rgb.shape}")

    # Save first few as files for visual inspection
    if i < 5:
        img = Image.fromarray(rgb)
        img.save(f"arcagi_state_{i}.png")
        print(f"    Saved: arcagi_state_{i}.png")

# ── Step 6: Test DINOv2 feature discrimination ──────────────────────────────
print("\n[6] Testing DINOv2 feature discrimination on game states...")

if len(rgb_images) >= 2:
    try:
        import torch
        from torchvision import transforms

        dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        dinov2.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dinov2 = dinov2.to(device)
        print(f"  DINOv2 loaded on {device}")

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        def encode_rgb(rgb_arr):
            img = Image.fromarray(rgb_arr)
            tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                features = dinov2.forward_features(tensor)
                cls_token = features["x_norm_clstoken"]
                cls_token = torch.nn.functional.normalize(cls_token, dim=-1)
            return cls_token

        # Encode all states
        features = []
        for i, rgb in enumerate(rgb_images):
            f = encode_rgb(rgb)
            features.append(f)
            print(f"  State {i} encoded: dim={f.shape[-1]}")

        # Pairwise cosine similarities
        print("\n  Pairwise cosine similarities:")
        sims = []
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                cos_sim = torch.nn.functional.cosine_similarity(
                    features[i], features[j]
                ).item()
                sims.append(cos_sim)
                print(f"    state_{i} vs state_{j}: cos_sim = {cos_sim:.4f}")

        avg_sim = np.mean(sims)
        min_sim = np.min(sims)
        max_sim = np.max(sims)
        print(f"\n  Average cos_sim: {avg_sim:.4f}")
        print(f"  Min cos_sim:     {min_sim:.4f}")
        print(f"  Max cos_sim:     {max_sim:.4f}")

        if avg_sim < 0.85:
            print("\n  ✓ DISCRIMINATIVE — DINOv2 distinguishes game states well!")
            print("  → This architecture should work for ARC-AGI-3.")
            print("  → ssl_ewa will be meaningful (unlike MiniWorld's 0.0001)")
            print("  → MPC can plan because different actions → different predictions")
        elif avg_sim < 0.95:
            print("\n  ~ MODERATE — some discrimination, may work with tuning")
            print("  → Consider using patch tokens instead of CLS token")
        else:
            print("\n  ✗ POOR — states too similar for CLS token")
            print("  → Same issue as MiniWorld")
            print("  → Try patch tokens or a grid-specific encoder")

    except Exception as e:
        print(f"  DINOv2 test failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("  Not enough grid states collected — see step 4 output above")
    print("  The env.step() return format may need adjustment")

# ── Step 7: Architecture compatibility summary ─────────────────────────────
print("\n" + "=" * 60)
print("ARCHITECTURE COMPATIBILITY SUMMARY")
print("=" * 60)

print(f"""
Environment: ARC-AGI-3 / ls20
Grid states collected: {len(grids)}
Action space: {[a.name for a in GameAction]} (discrete — MPC compatible)

For our A-B-M architecture:
  System A (encoder): DINOv2 ViT-B/14
    - Input: 2D grid → render to RGB → resize 224x224 → encode
    - Output: 768-dim L2-normalized CLS token

  World Model (VJEPAPredictor):
    - z_pred = Predictor(z_t, action_onehot)
    - N_ACTIONS = {len([a for a in GameAction])} ({', '.join(a.name for a in GameAction)})

  System B (MPC planner):
    - RandomShootingMPC with K=256 candidates, H=7 horizon
    - Goal: z_goal from a "solved" state (if available)

  Key advantage over MiniWorld:
    - Grid changes DRAMATICALLY between actions (not subtle 3D camera shifts)
    - DINOv2 CLS token should differentiate well
    - ssl_ewa should be >> 0.0001 (predictor actually learns dynamics)
""")

# Scorecard
try:
    scorecard = arc.get_scorecard()
    print(f"Scorecard: {scorecard}")
except Exception as e:
    print(f"Scorecard: {e}")
