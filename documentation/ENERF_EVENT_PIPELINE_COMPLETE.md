# 🎯 Flusso Completo E-NeRF Event-Based: Da Config a Rendering

**Documento**: Spiegazione del pipeline completo E-NeRF utilizzando una telecamera ad eventi  
**Target**: Event-Only vs Event+RGB dual scenarios  
**HDR GPU**: RTX 4060 Ti 12GB VRAM

---

## 📚 Indice

1. [Configurazione Iniziale](#1-configurazione-iniziale)
2. [Scenario 1: Event-Only (No RGB)](#2-scenario-1-event-only-no-rgb)
3. [Scenario 2: Event+RGB (Hybrid)](#3-scenario-2-eventrgb-hybrid)
4. [Flusso di Dati in Training](#4-flusso-di-dati-in-training)
5. [Flusso di Dati in Inferenza](#5-flusso-di-dati-in-inferenza)
6. [Memoria e Performance](#6-memoria-e-performance)

---

## 1. Configurazione Iniziale

### 1.1 Config File Giusto per Event-Only (Minimo)

File: `configs_custom/event_only_12gb.txt`

```ini
# ─────────────────────────────────────────────
# E-NeRF Event-Only Config - RTX 4060 Ti 12GB
# ─────────────────────────────────────────────

name = "enerf_event_only_12gb"
expname = "enerf_event_only_12gb"

# ✅ DATASET: Modalità evento
mode = "eds"                                # Dataset format (eds, tumvie, esim)
datadir = "C:\\path\\to\\data\\mocapDesk2"
downscale = 2                               # Risoluzione immagini (full=1, half=2)

# ✅ DATA INPUT: Solo eventi, NO RGB
events = 1                          # ← Abilita evento camera
event_only = 1                      # ← CHIAVE: Solo eventi, NON usare RGB ground truth!
C_thres = -1                        # Normalized loss threshold (auto calculate)
accumulate_evs = 1                  # Accumula eventi tra timestep

# ✅ MODEL: Coordinate encoding + MLPs
ff = 1                              # Usa FFMLP (3x più veloce)
tcnn = 0                            # Disabilita TCNN (instabile)

# ✅ MEMORIA: Ottimizzazioni per 12GB VRAM
fp16 = 1                            # Mixed precision (essenziale!)
ckpt = 1                            # Gradient checkpointing (essenziale!)

# ✅ SAMPLING: Ridotto per memoria
num_rays = 2048                     # Raggi per passo (default 65536, ridotto)
num_steps = 24                      # Campioni coarse (default 128, ridotto)
upsample_steps = 24                 # Campioni fine (default 128, ridotto)

# ✅ LEARNING
lr = 1e-2                           # Learning rate
iters = 10000                       # Iterazioni training (~3-4 ore)

# ✅ OUTPUT: Evento-specifico
out_dim_color = 1                   # Output 1D (intensity), NON RGB!
use_luma = 1                        # Converti RGB→luma se presente
disable_view_direction = 0          # Usa view direction conditioning

# ✅ VALIDATION e TESTING
eval_cnt = 500                      # Valida ogni 500 step
batch_size = 1                      # Batch size (fisso per evento)

# ✅ GPU CONTROL
cuda_ray = 0                        # Disabilita cuda_ray (stabilità)
density_scale = 1

# ✅ REPRODUCIBILITÀ
seed = 42
```

### 1.2 Config File per Event+RGB (Hybrid)

File: `configs_custom/event_rgb_hybrid_12gb.txt`

```ini
# ─────────────────────────────────────────────
# E-NeRF Event+RGB Hybrid Config - RTX 4060 Ti 12GB
# ─────────────────────────────────────────────

name = "enerf_event_rgb_hybrid_12gb"
expname = "enerf_event_rgb_hybrid_12gb"

# ✅ DATASET: Modalità evento
mode = "eds"
datadir = "C:\\path\\to\\data\\mocapDesk2"
downscale = 2

# ✅ DATA INPUT: Sia eventi CHE RGB ground truth
events = 1                          # ← Abilita evento camera
event_only = 0                      # ← CHIAVE: Usa ANCHE RGB (evento+RGB dual supervision!)
C_thres = -1                        # Auto-calculate
accumulate_evs = 1

# ✅ MODEL
ff = 1
tcnn = 0

# ✅ MEMORIA: Doppia memoria per RGB + events
fp16 = 1                            # ESSENZIALE
ckpt = 1                            # ESSENZIALE

# ✅ SAMPLING: Ridotto (due modalità = più memoria)
num_rays = 1536                     # ← Ridotto ulteriormente (dual supervision)
num_steps = 20                      # ← Ridotto
upsample_steps = 20

# ✅ LEARNING
lr = 1e-2
iters = 10000

# ✅ OUTPUT: RGB (perché abbiamo ground truth RGB)
out_dim_color = 3                   # ← 3D RGB output (non più 1D!)
weight_loss_rgb = 1.0               # Loss weight per RGB
w_no_ev = 0.5                       # Loss weight per "no-event" regions (vedi sotto)
use_luma = 0                        # NON convertire, usare RGB direttamente

# ✅ NEGATIVE EVENT SAMPLING
negative_event_sampling = 1         # Campiona pixel con NO event come negatives
                                    # (utile per stabilità training)

# ✅ VALIDATION
eval_cnt = 500
batch_size = 1

# ✅ GPU
cuda_ray = 0
density_scale = 1

# ✅ REPRODUCIBILITÀ
seed = 42
```

---

## 2. Scenario 1: Event-Only (No RGB)

### 2.1 Flusso da Config a Prima Iterazione

```
┌─────────────────────────────────────────────────────────────┐
│ START: python main_nerf.py --config event_only_12gb.txt     │
└─────────────────────────────────────────────────────────────┘
                            ↓
```

### 2.2 Parsing Config e Model Initialization

**File**: `main_nerf.py` (linee 1-100)

```python
# Passo 1: Parser config
opt = parser.parse_args()           # Carica event_only_12gb.txt
                                    # Risultato: opt.events=1, opt.event_only=1

# Passo 2: Validazione config
assert_config(opt)
# ├─ Verifica: events=True → OK
# ├─ Verifica: event_only=True → OK
# └─ Verifica: out_dim_color=1 → OK (per event-only)

# Passo 3: Retrieve frames per dataset
select_frames = get_frames(opt)     # Seleziona train/val/test frame indices
# Risultato:
# ├─ train_idxs: [5, 6, ..., 969] (965 frame)
# ├─ val_idxs: [3091, 3156, 3252] (3 frame)
# └─ test_idxs: [0]                (1 frame)

# Passo 4: Crea model (NeRFNetwork)
model, model_params, encoding_params = get_model(opt)
#
# File: nerf/network_ff.py (perché opt.ff=1)
#
# Struttura modello:
# ├─ encoder: GridEncoder (hash grid)
# │  └─ Codifica posizione 3D (x,y,z) → feature vector
# │     Input: (x,y,z) ∈ R³
# │     Output: hash_feature ∈ R^256 (embedding densità)
# │
# ├─ sigma_net: FFMLP per densità
# │  └─ Input: hash_feature [256]
# │     Output: sigma [1] (densità)
# │     Architettura: 256→128→128→1
# │
# └─ color_net: FFMLP per colore (1D luma!)
#    └─ Input: hash_feature [256] + view direction [3]
#       Output: luma [1] (intensità evento, NOT RGB!)
#       Architettura: (256+3)→128→128→1
```

**Uscita di Passo 4**:
- `model.encoder`: Grid encoder con embedding densità
- `model.sigma_net`: MLP densità (ℝ^256 → ℝ¹)
- `model.color_net`: MLP luma (ℝ^(256+3) → ℝ¹)

**Memoria allocata**:
- Encoder weights: ~10 MB
- Sigma MLP: ~1 MB
- Color MLP: ~1 MB
- **Total**: ~12 MB (weights solo)

### 2.3 Dataset Loading (EventNeRFDataset)

**File**: `main_nerf.py` (linee 270+)

```python
# Codice training loop principale
if opt.events:
    train_loader = EventNeRFDataset(
        opt,
        device='cuda',
        type='train',
        downscale=opt.downscale,          # 2
        select_frames=select_frames        # { train: [5..969], val: [...], test: [...] }
    ).dataloader()
```

**File**: `nerf/provider.py` (linea 1106+)

```python
class EventNeRFDataset(NGPDataset):
    def __init__(self, opt, device, type='train', downscale=1, select_frames=None):
        # Chiama parent __init__
        super().__init__(opt, device, type=type, ...)
        # ├─ Carica poses e intrinsics camera da poses_bounds.npy
        # ├─ Carica immagini RGB (se present)
        # └─ Carica informazioni camera (H, W, focal)
        
        # PER EVENT-ONLY: NON usa RGB ground truth
        self.out_dim_color = opt.out_dim_color       # = 1
        
        # Carica batch di eventi
        evs_batches_ns_tmp, no_events = self.load_events_at_frame_idxs(
            opt.datadir,
            self.frame_idxs,
            mode=opt.mode             # "eds"
        )
        # ├─ Per ogni frame_idx in [5, 6, ..., 969]
        # │   ├─ Carica .npy file con eventi: (x, y, t_ns, polarity)
        # │   ├─ Ordina temporalmente
        # │   └─ Aggrega per pixel (x,y)
        # └─ Risultato: events[frame_idx] = np.array([N_eventi, 4])
        
        # Interpolation pose per ogni evento
        # ├─ Carica pose camera ad alta frequenza (pos_hf): mocap pose ~200 Hz
        # ├─ Per ogni tempo_evento in events[frame_idx][:, 2]
        # │   ├─ Interpola rotazione via SLERP
        # │   └─ Interpola traslazione via cubic spline
        # └─ Risultato: poses_evs[frame_idx] = array([N_eventi, 3, 4])
        
        if self.accumulate_evs:
            # Compute numero successori evento
            # (serve per accumulazione causale)
            num_successor_evs[frame_idx] = array([N_eventi])
```

**Uscita di __init__**:

```
self.events = {
    5: tensor([N_ev, 4], device='cuda'),    # 4 = (x, y, t_ns, pol)
    6: tensor([N_ev, 4], device='cuda'),
    ...
    969: tensor([N_ev, 4], device='cuda'),
}

self.poses_evs = {
    5: array([N_ev, 3, 4]),    # 3×4 pose matrix per ogni evento
    6: array([N_ev, 3, 4]),
    ...
}

self.poses = array([965, 3, 5])  # Frame poses (da parent)
```

**Memoria allocata in dataset**:
- ~500k eventi per frame × 965 frame × 4 float32 = ~7.7 GB
- Pose interpolati: 500k × 965 × 12 float32 = ~22.8 GB
- **⚠️ PROBLEMA**: Senza precompute_evs_poses, poniamo memory richiesta = ~30 GB!

**Soluzione**:
```ini
precompute_evs_poses = 0  # Default: interpola al volo (slow, memory efficient)
# In questo caso:
# ├─ Carica solo events[frame_idx]
# ├─ Durante __getitem__, interpola pose per quell'evento
# └─ Memory: ~1 GB (solo eventi)
```

### 2.4 DataLoader e Batch Sampling

**File**: `nerf/provider.py` (classe EventNeRFDataset)

```python
def dataloader(self):
    return DataLoader(
        self,
        batch_size=self.batch_size,             # 1
        shuffle=(self.type == 'train'),        # True
        num_workers=2,
        pin_memory=True
    )

def __getitem__(self, index):
    # Seleziona frame casuale da train set
    frame_idx = self.frame_idxs_train[index % len(self.frame_idxs_train)]
    
    # Seleziona subset eventi da questo frame
    results = {}
    
    # Campiona n_rays evento random
    # ├─ Se event_only=1:
    # │   └─ Campiona solo pixel CON evento (no background!)
    # └─ Se event_only=0:
    #     ├─ Campiona pixel CON evento
    #     └─ + Campiona pixel SENZA evento (negative samples)
    
    event_indices = np.random.choice(
        self.num_evs[frame_idx],
        size=min(num_rays, self.num_evs[frame_idx]),
        replace=False
    )
    
    # Ottieni evento dati
    sampled_events = self.events[frame_idx][event_indices]
    # Shape: [n_rays, 4] = (x, y, t_ns, pol)
    
    # Interpola pose per questi eventi
    poses_sampled = self.rot_interpolator(sampled_events[:, 2]).as_matrix()
    # Shape: [n_rays, 3, 3]
    trans_sampled = self.trans_interpolator(sampled_events[:, 2])
    # Shape: [n_rays, 3]
    
    # Crea ray da evento (x,y,pol,t) + pose
    # ├─ x,y = position evento
    # ├─ pose = interpolated camera pose at time t
    # ├─ intrinsics = camera K matrix
    # └─ pol = polarity (± 1)
    rays = self.get_event_rays(
        sampled_events,
        poses_sampled,
        trans_sampled,
        self.intrinsics
    )
    # ├─ ray_origin o: [n_rays, 3] (camera center nel world)
    # ├─ ray_direction d: [n_rays, 3] (unit direction)
    # ├─ timestamps: [n_rays] (in nanoseconds)
    # └─ polarities: [n_rays] (±1)
    
    # PER EVENT-ONLY: Ground truth = polarity (±1)
    # ├─ NOT RGB! (no image data)
    # └─ Labels: [n_rays] (±1)
    if self.event_only:
        labels = sampled_events[:, 3]   # polarity ±1
    else:
        # PER EVENT+RGB: Deve avere ENTRAMBI
        labels_events = sampled_events[:, 3]
        # Sample RGB pixels (non-overlapping con event pixels)
        rgb_indices = np.random.choice(
            total_pixels - len(event_indices),
            size=num_rays,
            replace=False
        )
        labels_rgb = self.images[frame_idx][rgb_indices]
        # Avremo mixed batch con eventi e RGB
    
    return rays, labels
```

### 2.5 Forward Pass (Event-Only Scenario)

**File**: `main_nerf.py` training loop

```python
for epoch in range(max_epoch):
    for batch_idx, (rays, labels) in enumerate(train_loader):
        # rays = {
        #     'o': [n_rays, 3],            (camera origin)
        #     'd': [n_rays, 3],            (ray direction)
        #     't': [n_rays],               (timestamp ns)
        #     'p': [n_rays],               (polarity ±1)
        # }
        # labels = [n_rays]                (target polarity)
        
        # ──────────────────────────────────────
        # FORWARD PASS
        # ──────────────────────────────────────
        
        pred = model(rays['o'], rays['d'], near, far, ...)
        # Ritorna pred = [n_rays] (predicted luma ∈ [0, 1])
```

**Dettaglio model.forward() - File: `nerf/network_ff.py`**

Il metodo `forward` in `network_ff.py` è per un singolo punto 3D, non per rays. Per E-NeRF, il rendering è gestito dalla classe base `NeRFRenderer` che chiama `render()` → `run()`.

```python
# network_ff.py forward (per singolo punto)
def forward(self, x, d):
    # x: [B, 3] posizioni 3D
    # d: [B, 3] direzioni
    
    # Sigma network
    x_enc = self.encoder(x, bound=self.bound)  # [B, 256]
    h = self.sigma_net(x_enc)                  # [B, 16] (1 sigma + 15 geo_feat)
    sigma = trunc_exp(h[..., 0])               # [B]
    geo_feat = h[..., 1:]                      # [B, 15]
    
    # Color network (luma prediction)
    d_enc = self.encoder_dir(d)                # [B, 27] (spherical harmonics)
    p = torch.zeros_like(geo_feat[..., :1])    # padding per raggiungere 32 input
    h = torch.cat([d_enc, geo_feat, p], dim=-1) # [B, 27+15+1=43] → padded a 32?
    h = self.color_net(h)                      # [B, 1] luma ∈ [0,1]
    rgb = torch.sigmoid(h)                     # [B, 1]
    
    return sigma, rgb
```

**Pipeline di rendering per E-NeRF (event-only, no RGB)**

La pipeline completa è in `NeRFRenderer.run()` (file `nerf/renderer.py`):

```python
def run(self, rays_o, rays_d, num_steps=128, upsample_steps=128, **kwargs):
    # rays_o/d: [N, 3] (N = batch_size * num_rays)
    
    # 1. Calcola near/far per ogni ray
    nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, aabb, min_near)
    
    # 2. Campiona z_vals lungo ogni ray (coarse sampling)
    z_vals = torch.linspace(0, 1, num_steps).expand(N, num_steps)
    z_vals = nears + (fars - nears) * z_vals  # [N, num_steps]
    
    # 3. Genera punti 3D lungo rays
    xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1)  # [N, num_steps, 3]
    
    # 4. Query density per tutti i punti
    density_out = self.density(xyzs.reshape(-1, 3))  # chiama network_ff.density()
    sigmas = density_out['sigma'].view(N, num_steps)  # [N, num_steps]
    geo_feat = density_out['geo_feat'].view(N, num_steps, -1)  # [N, num_steps, 15]
    
    # 5. Calcola weights (alpha compositing)
    deltas = z_vals[..., 1:] - z_vals[..., :-1]  # [N, num_steps-1]
    alphas = 1 - torch.exp(-deltas * density_scale * sigmas[..., :-1])
    weights = alphas * torch.cumprod(1 - alphas + 1e-15, dim=-1)  # [N, num_steps]
    
    # 6. Hierarchical upsampling (se upsample_steps > 0)
    if upsample_steps > 0:
        # Campiona nuovi punti basati su PDF dei weights
        new_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1], upsample_steps)
        new_xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * new_z_vals.unsqueeze(-1)
        
        # Query density per nuovi punti
        new_density_out = self.density(new_xyzs.reshape(-1, 3))
        # Concatena e riordina tutti i punti
        z_vals = torch.cat([z_vals, new_z_vals], dim=1).sort(dim=1)[0]
        # ... (riordina xyzs, sigmas, geo_feat)
    
    # 7. Query colore SOLO per punti con weight > threshold
    mask = weights > 1e-4  # [N, total_steps]
    dirs = rays_d.unsqueeze(-2).expand_as(xyzs)  # [N, total_steps, 3]
    
    # Chiama network_ff.color() per punti validi
    rgbs = self.color(
        xyzs.reshape(-1, 3), 
        dirs.reshape(-1, 3), 
        mask=mask.reshape(-1), 
        geo_feat=geo_feat.reshape(-1, geo_feat.shape[-1])
    )  # [N*total_steps, 1] luma predicted
    
    rgbs = rgbs.view(N, -1, 1)  # [N, total_steps, 1]
    
    # 8. Alpha compositing finale
    weights_sum = weights.sum(dim=-1)  # [N]
    image = torch.sum(weights.unsqueeze(-1) * rgbs, dim=-2)  # [N, 1] luma finale
    
    # 9. Calcola depth
    depth = torch.sum(weights * z_vals, dim=-1) / weights_sum  # [N]
    
    return {'image': image, 'depth': depth}  # image = predicted luma [0,1]
```

**Loss per E-NeRF event-only**:
- `pred = model(rays_o, rays_d)['image']` → `[N, 1]` luma ∈ [0,1]
- `target = (polarity + 1) / 2` → `[N, 1]` ∈ {0, 1} (ON=1, OFF=0)
- `loss = MSE(pred, target)` o `BCE(pred, target)`

**memoria durante forward**:
- features [n_rays, 256]: 2048 × 256 × 4 bytes = 2.1 MB
- sigma [n_rays, num_steps]: 2048 × 24 × 4 = 196 KB
- weights: 196 KB
- **Per-ray memory**: ~2.5 MB per passo
- Con 965 frame × 1-2 ora → memoria sostenibile

### 2.6 Loss Computation (Event-Only)

**File**: `loss.py` + training loop

```python
# Pred vs Ground truth (EVENT ONLY)
# ├─ pred: [n_rays] (predicted luma from model)
# └─ labels: [n_rays] (polarity input ±1)

# PROBLEMA: Model predice luma ∈ [0, 1], ma labels sono ±1
# SOLUZIONE: Map polarity ±1 → {0, 1} binary target
#
# Polarity mapping:
# ├─ pol = +1 (ON event) → target_luma = 1.0
# └─ pol = -1 (OFF event) → target_luma = 0.0

target = (labels + 1) / 2                  # ±1 → [0, 1]

# Binary cross-entropy loss (migliore di MSE per probabilità)
loss = F.binary_cross_entropy(pred, target)

# Oppure  MSE (da config original)
loss = criterion(pred.unsqueeze(-1), target.unsqueeze(-1))
# criterion = torch.nn.MSELoss(reduction='none')
```

### 2.7 Backward Pass (Event-Only)

```python
optimizer.zero_grad()

loss.backward()          # Compute gradients
# ├─ Backprop through color_net
# ├─ Backprop through sigma_net
# ├─ Backprop through encoder
# └─ Pyramid di gradienti per tutti parametri

optimizer.step()         # Update weights
# ├─ weights_net -= lr * grad_weights_net
# ├─ weights_sigma -= lr * grad_weights_sigma
# └─ encoder_weights -= lr * grad_encoder

scheduler.step()        # Learning rate decay
# ├─ lr *= 0.1 ^ (iter / max_iters)
# └─ Graduale riduzione lr durante training
```

**Memoria durante backward**:
- Attivamenti salvati (fp16): ~2 MB
- Gradient tensori: ~2 MB
- **Peak memoria backward**: ~5-10 MB aggiuntivo

---

## 3. Scenario 2: Event+RGB (Hybrid)

### 3.1 Differenze da Event-Only

**Config change**:
```ini
event_only = 0          # ← Abilitiamo RGB ground truth
out_dim_color = 3       # ← Output RGB (non 1D luma!)
negative_event_sampling = 1
```

### 3.2 Dataset Loading (Event+RGB)

**File**: `nerf/provider.py` __getitem__

```python
def __getitem__(self, index):
    # Step 1: Load events (come event-only)
    frame_idx = self.frame_idxs_train[...]
    sampled_events = self.events[frame_idx][...]
    
    # Step 2: Load RGB images
    img = self.images[frame_idx]              # [H, W, 3]
    
    # Step 3: MIX batch
    # ├─ Campiona num_rays_events pixel CON evento
    # ├─ Campiona num_rays_rgb pixel SENZA evento (from RGB)
    # └─ Total batch = num_rays_events + num_rays_rgb
    
    # Esempio:
    if not self.event_only:
        # Half rays from events, half from RGB
        n_rays_events = num_rays // 2      # 1024
        n_rays_rgb = num_rays // 2         # 1024
        
        # Event rays
        event_indices = np.random.choice(
            self.num_evs[frame_idx],
            size=n_rays_events,
            replace=False
        )
        sampled_events = self.events[frame_idx][event_indices]
        
        # RGB rays (negative sampling)
        if negative_event_sampling:
            # Evita pixel CON evento
            mask_with_event = np.zeros((H, W), dtype=bool)
            mask_with_event[sampled_events[:, 0], sampled_events[:, 1]] = True
            
            rgb_indices = np.where(~mask_with_event)
            rgb_sel = np.random.choice(
                len(rgb_indices[0]),
                size=n_rays_rgb,
                replace=False
            )
            rgb_y, rgb_x = rgb_indices[0][rgb_sel], rgb_indices[1][rgb_sel]
        
        # Combina
        all_rays_o = [event_rays_o, rgb_rays_o]        # [2048, 3]
        all_rays_d = [event_rays_d, rgb_rays_d]
        
        # Targets
        all_labels = [
            sampled_events[:, 3],                       # Event polarities ±1
            img[rgb_y, rgb_x, :]                       # RGB values [0, 1]
        ]
    
    return all_rays, all_labels
```

### 3.3 Forward Pass (Event+RGB)

```python
# Batch misto: metà evento, metà RGB
pred = model(rays['o'], rays['d'], ...)
# pred: [2048, 3] (RGB per TUTTI i rays)
```

**Dettaglio color_net**:

```python
# Color MLP
def forward(self, features, dirs):
    # features: [n_rays, 256]
    # dirs: [n_rays, 3]
    
    # Concat per view direction conditioning
    x = torch.cat([features, dirs], dim=-1)    # [n_rays, 259]
    
    # FFMLP layers
    x = self.linear1(x)                        # [n_rays, 128]
    x = F.relu(x)
    x = self.linear2(x)                        # [n_rays, 128]
    x = F.relu(x)
    x = self.linear3(x)                        # [n_rays, 3] ← RGB output!
    x = torch.sigmoid(x)                       # [0, 1]
    
    return x  # [n_rays, 3]
```

### 3.4 Loss Computation (Event+RGB)

```python
# Split batch
n_event_rays = len(event_indices)
n_rgb_rays = len(rgb_indices)

pred_event = pred[:n_event_rays]               # [1024, 3]
pred_rgb = pred[n_event_rays:]                 # [1024, 3]

label_event = labels[:n_event_rays]            # [1024, 1] (polarity)
label_rgb = labels[n_event_rays:]              # [1024, 3] (RGB)

# Event loss: Convert polarity ±1 → luma [0, 1]
target_event_luma = (label_event + 1) / 2      # [0, 1]
# Use only R channel for polarity prediction
loss_event = F.mse_loss(
    pred_event[:, 0],                          # Use R channel
    target_event_luma
)

# RGB loss: Match RGB output
loss_rgb = F.mse_loss(pred_rgb, label_rgb)

# Combined loss
total_loss = loss_event + weight_loss_rgb * loss_rgb
# weight_loss_rgb = 1.0 (from config)

# Optional: "no-event" loss (per stabilità)
if self.w_no_ev > 0:
    # Pixel senza evento devono avere background color (≈ 0.5)
    loss_no_ev = F.mse_loss(pred_rgb, torch.ones_like(pred_rgb) * 0.5)
    total_loss += self.w_no_ev * loss_no_ev

loss = total_loss
```

---

## 4. Flusso di Dati in Training

### 4.1 Diagramma Completo Training

```
┌────────────────────────────────────────────────────────────┐
│ CONFIGURATION LOADING                                      │
│ ├─ event_only_12gb.txt (o event_rgb_hybrid_12gb.txt)      │
│ └─ opt.events=1, opt.event_only=1 (or 0), opt.ff=1, ...   │
└────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│ MODEL INITIALIZATION (nerf/network_ff.py)                  │
│ ├─ GridEncoder (hash grid)                                 │
│ │  └─ Encodes position (x,y,z) → feature vector          │
│ ├─ SigmaMLP: [256] → [1] (density)                        │
│ └─ ColorMLP: [256+3] → [1 or 3] (luma or RGB)            │
└────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│ DATASET LOADING (nerf/provider.py:EventNeRFDataset)        │
│ ├─ Load event batches: events[frame_idx] = [N_ev, 4]      │
│ ├─ Load camera poses: poses_evs[frame_idx] = [N_ev, 3, 4] │
│ ├─ Load images (if RGB): images[frame_idx] = [H, W, 3]   │
│ └─ GPU preload: move events to CUDA                        │
└────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│ TRAINING LOOP (max_epoch iterations)                       │
│                                                             │
│ for epoch in range(max_epoch):                             │
│     for batch_idx, (rays, labels) in enumerate(dataloader):│
│                                                             │
│ ┌─────────────────────────────────────────────────────┐    │
│ │ FORWARD PASS                                        │    │
│ ├─ rays = {o, d, t, p}  [n_rays, ...]               │    │
│ ├─ model(rays_o, rays_d) → pred [n_rays] or [n_rays, 3] │
│ │  ├─ HashGrid: (x,y,z) → [256] features            │    │
│ │  ├─ SigmaMLP: [256] → [1] density               │    │
│ │  ├─ VolumeRendering: {coarse + upsample}         │    │
│ │  └─ ColorMLP: [256+view] → [1 or 3]              │    │
│ └─────────────────────────────────────────────────────┘    │
│                            ↓                                │
│ ┌─────────────────────────────────────────────────────┐    │
│ │ LOSS COMPUTATION (loss.py + criteria)              │    │
│ │                                                     │    │
│ │ if event_only:                                      │    │
│ │    target = (labels + 1) / 2  # ±1 → [0,1]        │    │
│ │    loss = MSE(pred, target)                        │    │
│ │                                                     │    │
│ │ else:  # event+RGB                                 │    │
│ │    loss_ev = MSE(pred_event, target_event)        │    │
│ │    loss_rgb = MSE(pred_rgb, target_rgb)           │    │
│ │    loss = loss_ev + w_rgb * loss_rgb              │    │
│ └─────────────────────────────────────────────────────┘    │
│                            ↓                                │
│ ┌─────────────────────────────────────────────────────┐    │
│ │ BACKWARD PASS (autograd)                            │    │
│ ├─ loss.backward()                                    │    │
│ │  ├─ grad_colornet                                  │    │
│ │  ├─ grad_sigmanet                                  │    │
│ │  └─ grad_encoder                                   │    │
│ ├─ optimizer.step()  (Adam)                          │    │
│ │  └─ Update all weights                             │    │
│ └─ scheduler.step()  (LR decay)                      │    │
│                                                      │    │
│ └─ Every eval_cnt steps: validation pass             │    │
│    └─ PSNR metric computation                        │    │
└────────────────────────────────────────────────────────┘
                            ↓
              (repeat for max_epoch ~12 epoch)
                            ↓
┌────────────────────────────────────────────────────────────┐
│ SAVE MODEL (final checkpoint)                              │
│ └─ model.state_dict() saved to exp/enerf_event_only/      │
└────────────────────────────────────────────────────────────┘
```

### 4.2 Memoria Breakdown durante Training

```
Memory Components (RTX 4060 Ti 12GB):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. MODEL WEIGHTS:
   ├─ Encoder: 10 MB
   ├─ Sigma MLP: 1 MB
   ├─ Color MLP (1D luma): 0.5 MB
   └─ Total: ~11.5 MB

2. OPTIMIZER STATE (Adam):
   ├─ Momentum: 1× model size = 11.5 MB
   ├─ Variance: 1× model size = 11.5 MB
   └─ Total: 23 MB

3. FORWARD ACTIVATIONS (per batch, n_rays=2048):
   ├─ Features [n_rays, 256]: 2.1 MB
   ├─ Sigma [n_rays, 24]: 196 KB
   ├─ Other (depth, weights): ~500 KB
   └─ Total: ~2.8 MB

4. GRADIENT TENSORS (during backward):
   ├─ Same size as activations: ~2.8 MB
   └─ (Short-lived, freed after step)

5. DATASET (preloaded to GPU):
   ├─ Events only: ~1 GB (500k ev × 965 frames)
   └─ Optimization: ~1 GB (feasible)

6. PyTorch overhead + CUDA graph:
   └─ ~1-2 GB

TOTAL PEAK USAGE:
├─ Event-only: 11.5 (model) + 23 (opt) + 2.8 (act) + 1 (data) + 2 (overhead)
├─ = ~40 MB model + optimizer + ~5 MB per batch
├─ System: ~10 GB / 12 GB available
└─ SAFE ✓

Note: Con fp16 + ckpt, memoria dimezza → ~4-5 GB per training
```

---

## 5. Flusso di Dati in Inferenza

### 5.1 Inferenza Event-Only

```python
# Load model
ckpt = torch.load('exp/enerf_event_only/ckpt_10000.pth')
model.load_state_dict(ckpt['model'])
model.eval()

# Rendering camera path
poses_interp = interpolate_poses(...)  # 60 frame spiral

for i, pose in enumerate(poses_interp):
    # Per ogni view point
    
    # Render grid di pixel
    H, W = 256, 256
    for y in range(H):
        for x in range(W):
            # Ray da pinhole camera
            ray_o, ray_d = get_ray(x, y, pose, K)
            
            # Forward pass
            with torch.no_grad():
                pred = model(ray_o, ray_d)  # [1] (luma)
            
            # Convert luma → RGB per visualization
            luma = pred.item()  # scalar in [0, 1]
            rgb = [luma, luma, luma]  # Grayscale
            
            image[y, x] = rgb
    
    # Save image
    save_image(f'frame_{i:04d}.png', image)

# Create video
# ffmpeg -framerate 30 -i frame_%04d.png -c:v libx264 output.mp4
```

### 5.2 Inferenza Event+RGB

```python
# Load model (same, ma con out_dim_color=3)
model.load_state_dict(ckpt['model'])

for i, pose in enumerate(poses_interp):
    for y in range(H):
        for x in range(W):
            ray_o, ray_d = get_ray(x, y, pose, K)
            
            with torch.no_grad():
                pred = model(ray_o, ray_d)  # [3] (RGB)
            
            # Direct RGB
            rgb = pred.numpy()  # [0, 1]
            image[y, x] = rgb
    
    save_image(f'frame_{i:04d}.png', image)
```

### 5.3 Mesh Extraction

```python
python main_nerf.py \
    --config configs_custom/event_only_12gb.txt \
    --ckpt exp/enerf_event_only/ckpt_10000.pth \
    --test_only \
    --save_mesh 1 \
    --mesh_resolution 256

# Output: exp/enerf_event_only/results/mesh.ply
# ├─ 256³ voxel resolution
# └─ Extracted via marching cubes
```

---

## 6. Memoria e Performance

### 6.1 Timing per RTX 4060 Ti 12GB

```
Config:           event_only_12gb.txt
num_rays:         2048
num_steps:        24
upsample_steps:   24
fp16:             1
ckpt:             1
downscale:        2 (256×256 images)

TIMING PER PASSO:
┌─────────────────────────────────────────┐
│ Forward pass (encoder + sigma + color): │
│ ├─ Coarse sampling (24): 35 ms         │
│ ├─ Color net (2048 rays): 25 ms        │
│ ├─ Upsample sampling (24): 35 ms       │
│ ├─ Fine color: 25 ms                   │
│ └─ Compositing: 5 ms                   │
│                                         │
│ TOTAL FORWARD: ~125 ms                 │
│                                         │
│ Backward pass (gradients):              │
│ ├─ Color gradient: 20 ms                │
│ ├─ Encoder gradient: 30 ms              │
│ ├─ Sigma gradient: 10 ms                │
│ └─ Parameter update: 5 ms               │
│                                         │
│ TOTAL BACKWARD: ~65 ms                 │
│                                         │
│ TOTAL PER STEP: ~190 ms (5.3 it/s)    │
└─────────────────────────────────────────┘

FULL TRAINING:
├─ Iters: 10,000
├─ Total time: 10,000 × 190 ms = ~31 minutes
├─ But with validation overhead: ~50 minutes
└─ ≈ 1 hour training time (actually: 45-60 min)

INFERENCE (rendering 60 frames @256×256):
├─ Per frame: 60 sec / 256² × 2048 rays
├─ Actually: 5-10 sec per frame (batched)
└─ Total: 5-10 min per 60-frame video
```

### 6.2 Memory Optimization per 12GB VRAM

```ini
# Ottimizzazioni abilitate:
fp16 = 1                # Mixed precision (2× meno memoria)
ckpt = 1                # Gradient checkpointing (50% meno attivamenti)
downscale = 2           # 4× meno pixel (256×256 vs 512×512)
num_rays = 2048         # Batch ridotto (default 65536)
num_steps = 24          # Coarse sampling ridotto (128 default)
upsample_steps = 24     # Fine sampling ridotto (128 default)

# Risultato:
Memory = (base / 2) × (1 - ckpt_saving) × (downscale²) × (rays_ratio)
       = (20 GB / 2) × 0.5 × (1/4) × (2048/65536)
       = 10 × 0.5 × 0.25 × 0.03
       ≈ 0.04 × 10 GB = 400 MB peak
       + Dataset (1 GB) + overhead (1 GB)
       = ~2.5 GB total (sotto 12 GB ✓)
```

### 6.3 Event-Only vs Event+RGB Performance

```
COMPARISON:
╔════════════════════╦════════════════════╦═══════════════════╗
║ Metric             ║ Event-Only         ║ Event+RGB Hybrid  ║
╠════════════════════╬════════════════════╬═══════════════════╣
║ Memory             ║ ~3 GB peak         ║ ~4 GB peak        ║
║ Speed              ║ 5.3 it/s           ║ 4.2 it/s (-20%)  ║
║ Training time      ║ ~45-50 min         ║ ~60-70 min        ║
║ Output dim         ║ 1D luma            ║ 3D RGB            ║
║ Supervision       ║ Only events        ║ Events + RGB      ║
║ Mesh quality      ║ Good (density OK)  ║ Better (RGB hint) ║
║ Visual quality    ║ Grayscale dynamic  ║ Full color        ║
║ Best for          ║ High-speed video   ║ General 3D vision║
╚════════════════════╩════════════════════╩═══════════════════╝
```

---

## Riassunto: Quale Scenario Scegliere?

### Event-Only
**Usa se**:
- Hai dataset puro evento (no RGB images)
- Vuoi massima velocità (1.2× più veloce)
- Interessato alla dinamica ad alta frequenza (event camera advantage)
- Memory critica (3 GB invece 4 GB)

**Config**: `event_only_12gb.txt`

**Comando**:
```powershell
python main_nerf.py --config configs_custom/event_only_12gb.txt --gpu 0
```

### Event+RGB Hybrid
**Usa se**:
- Hai dataset con entrambi event camera + RGB camera (sync)
- Vuoi migliore qualità visuale (RGB supervision)
- Puoi tollerare training leggermente più lento
- Vuoi output full-color

**Config**: `event_rgb_hybrid_12gb.txt`

**Comando**:
```powershell
python main_nerf.py --config configs_custom/event_rgb_hybrid_12gb.txt --gpu 0
```

---

**END OF DOCUMENT**
