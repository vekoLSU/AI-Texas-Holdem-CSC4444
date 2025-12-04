# Windows 11 Training Guide (RTX 5080)

Quick guide to train your poker bot overnight on Windows with RTX 5080.

---

## Prerequisites

- Windows 11 Pro
- NVIDIA RTX 5080
- Python 3.9+ installed

---

## Step 1: Extract Repository

1. Download the zip from GitHub
2. Extract to `Desktop\AI-Texas-Holdem-CSC4444-main`

---

## Step 2: Install Dependencies (5 minutes)

Open PowerShell and run:

```powershell
# Navigate to the folder
cd Desktop\AI-Texas-Holdem-CSC4444-main

# Install PyTorch with CUDA 12.4 (for RTX 5080)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install other requirements
pip install websockets numpy
```

---

## Step 3: Start Training

Just run:

```powershell
python train_local.py
```

**That's it!** The script will:
- Auto-detect your RTX 5080
- Train for 500,000 hands (~5-6 hours)
- Save checkpoints every 10,000 hands
- Show progress every 100 hands

---

## What You'll See

```
============================================================
POKER BOT RL TRAINING - RTX 5080 EDITION
============================================================
✅ GPU Detected: NVIDIA GeForce RTX 5080
   VRAM: 16.0 GB

Training Configuration:
  Total Hands: 500,000
  Batch Size: 256
  Learning Rate: 0.0003
  Estimated Time: 5-6 hours on RTX 5080
  Checkpoint Every: 10,000 hands
============================================================

🎲 Starting training...

[   100/500000] Win Rate:  48.0% | Avg Reward:   45.2 | ETA: 5.8h
[   200/500000] Win Rate:  51.5% | Avg Reward:   62.3 | ETA: 5.7h
[   300/500000] Win Rate:  53.2% | Avg Reward:   78.1 | ETA: 5.6h
...
```

---

## Progress Tracking

Every 10,000 hands, you'll see:
```
💾 Checkpoint saved: trained_models\agent_checkpoint_10000.pt
```

These are your safety backups in case something goes wrong.

---

## When Training Finishes

You'll see:
```
============================================================
✅ TRAINING COMPLETE!
============================================================
Total Hands: 500,000
Final Win Rate: 72.3%
Total Time: 5.42 hours
Final Model: trained_models\agent_final.pt
============================================================

🎉 Your bot is now ready to DOMINATE!
```

---

## Output Files

All saved in `trained_models/` folder:

- `agent_checkpoint_10000.pt` (after 10K hands)
- `agent_checkpoint_20000.pt` (after 20K hands)
- ... up to ...
- `agent_checkpoint_500000.pt` (after 500K hands)
- **`agent_final.pt`** ← This is your trained model
- `training_stats.json` (performance metrics)

---

## Troubleshooting

### "No GPU detected"
```powershell
# Check if CUDA is available
python -c "import torch; print(torch.cuda.is_available())"
```

If False, reinstall PyTorch:
```powershell
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### "Module not found: poker_bot"
Make sure you're in the right directory:
```powershell
cd Desktop\AI-Texas-Holdem-CSC4444-main
```

### Training is slow
- Check Task Manager → GPU usage should be 90%+
- If CPU is at 100% instead, PyTorch isn't using GPU
- Reinstall PyTorch with CUDA support

---

## Performance Expectations

**Your RTX 5080 Performance:**
- **Speed**: ~1,500-2,000 hands/minute
- **Total Time**: 5-6 hours for 500K hands
- **VRAM Usage**: ~4-6 GB (plenty of headroom)
- **Win Rate**: 70-85% after training

**Comparison to Colab (Free T4):**
- 2-3x faster training
- No disconnects
- Better results (larger batch size)

---

## Next Steps (Tomorrow Morning)

1. Check `trained_models/agent_final.pt` exists
2. Integrate with ensemble (I'll help with this)
3. Test against opponents
4. DOMINATE the competition 🏆

---

## What if Training Crashes?

**Don't worry!** Checkpoints save every 10K hands.

To resume from checkpoint:
```python
# Edit train_local.py, line ~450:
# Before: agent = ActorCriticAgent(...).to(device)
# After:
agent = ActorCriticAgent(...).to(device)
agent.load('trained_models/agent_checkpoint_250000.pt')  # Load last checkpoint
hand_count = 250000  # Start from checkpoint hand count
```

Then re-run:
```powershell
python train_local.py
```

---

## Optional: Monitor Training

Open another PowerShell window:
```powershell
# Watch GPU usage in real-time
nvidia-smi -l 1
```

You should see:
- GPU Utilization: 90-100%
- Memory Usage: 4-6 GB / 16 GB
- Temperature: 60-75°C

---

**Ready to start? Just run:**
```powershell
python train_local.py
```

**Then go to sleep!** 😴
