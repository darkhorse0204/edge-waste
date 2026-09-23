"""DCGAN-based synthetic data augmentation for minority classes.

The current 33-class taxonomy has no meaningful imbalance to fix — 27 of its
33 classes hold exactly 500 images each. The class this module actually
targets is the professor's own 14-class custom dataset (still on disk under
`data/processed.OLD_14CLASS_STALE.bak/`), which does have real, uneven
counts: `textile` at 272 images against 666 for most other classes, `syringe`
at 562, `iv_bag` at 581. GAN augmentation only earns its complexity where a
genuine scarcity exists — applying it to an already-balanced class would
just be synthesising redundant, denser coverage of a distribution the real
data already represents well.

A small unconditional DCGAN, trained per-class, not a large or conditional
architecture: at a few hundred real training images, a bigger generator
would overfit long before learning a useful data distribution, and a single
minority class needs only one generator, not a shared conditional model
across classes with wildly different visual structure (textiles vs.
syringes) fighting the same weights.

Every generated image is written into its own `synthetic/` subfolder with a
`gan_` filename prefix — never mixed into the same folder as real images
indistinguishably — so provenance stays auditable and no downstream script
can accidentally treat a synthetic image as ground truth without a
deliberate opt-in.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LATENT_DIM = 100


class Generator(nn.Module):
    """Latent vector -> image_size x image_size x 3, via transposed convs.

    Sized for `image_size` in {32, 64} — a few hundred real images cannot
    support a generator built for ImageNet-scale resolution regardless of
    architecture, so this stays deliberately small rather than pretending a
    bigger network would help with this little data.
    """

    def __init__(self, latent_dim: int = LATENT_DIM, image_size: int = 64, channels: int = 3):
        super().__init__()
        assert image_size in (32, 64), "sized for 32 or 64; a larger target needs more real data first"
        base = 64
        layers = [
            nn.ConvTranspose2d(latent_dim, base * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(base * 8), nn.ReLU(True),
        ]
        cur, spatial = base * 8, 4
        while spatial < image_size // 2:
            nxt = max(base, cur // 2)
            layers += [
                nn.ConvTranspose2d(cur, nxt, 4, 2, 1, bias=False),
                nn.BatchNorm2d(nxt), nn.ReLU(True),
            ]
            cur, spatial = nxt, spatial * 2
        layers += [nn.ConvTranspose2d(cur, channels, 4, 2, 1, bias=False), nn.Tanh()]
        self.net = nn.Sequential(*layers)
        self.latent_dim = latent_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z.view(z.size(0), self.latent_dim, 1, 1))


class Discriminator(nn.Module):
    def __init__(self, image_size: int = 64, channels: int = 3):
        super().__init__()
        base = 64
        layers = [
            nn.Conv2d(channels, base, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        cur, spatial = base, image_size // 2
        while spatial > 4:
            nxt = cur * 2
            layers += [
                nn.Conv2d(cur, nxt, 4, 2, 1, bias=False),
                nn.BatchNorm2d(nxt), nn.LeakyReLU(0.2, inplace=True),
            ]
            cur, spatial = nxt, spatial // 2
        layers += [nn.Conv2d(cur, 1, 4, 1, 0, bias=False), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)


class _ImageFolderFlat(Dataset):
    """Every image directly under `root` — no canonical-class subfolders,
    since this trains one GAN per already-known class."""

    def __init__(self, root: Path, image_size: int):
        self.paths = sorted(p for p in root.iterdir()
                             if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        if not self.paths:
            raise ValueError(f"No images found under {root}")
        self.tfm = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),  # match Tanh's [-1, 1] output range
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        return self.tfm(Image.open(self.paths[i]).convert("RGB"))


def train_gan(
    class_dir: str | Path, output_dir: str | Path, image_size: int = 64,
    epochs: int = 100, batch_size: int = 32, lr: float = 2e-4, device: str = "cpu",
) -> dict:
    """Train one DCGAN on every image directly under `class_dir`.

    Returns a small history dict (loss curves) rather than nothing, so a
    caller can tell a genuinely-training GAN from one stuck at the
    well-known G/D loss equilibrium failure mode (both losses flat near
    ln(2) ~ 0.69 from the first epoch is mode collapse or a learning-rate
    mismatch, not slow convergence) without re-deriving it from the raw log.
    """
    device_t = torch.device(device)
    ds = _ImageFolderFlat(Path(class_dir), image_size)
    loader = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True, drop_last=True)

    gen = Generator(LATENT_DIM, image_size).to(device_t)
    disc = Discriminator(image_size).to(device_t)
    opt_g = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    history = {"g_loss": [], "d_loss": []}
    fixed_noise = torch.randn(16, LATENT_DIM, device=device_t)

    print(f"Training DCGAN on {len(ds)} real images from {class_dir} "
          f"({image_size}x{image_size}, {epochs} epochs, device={device})")
    for epoch in range(1, epochs + 1):
        g_losses, d_losses = [], []
        for real in loader:
            real = real.to(device_t)
            bs = real.size(0)
            real_labels = torch.full((bs,), 0.9, device=device_t)  # label smoothing
            fake_labels = torch.zeros(bs, device=device_t)

            # --- Discriminator ---
            opt_d.zero_grad()
            d_real = disc(real)
            loss_d_real = criterion(d_real, real_labels)
            noise = torch.randn(bs, LATENT_DIM, device=device_t)
            fake = gen(noise)
            d_fake = disc(fake.detach())
            loss_d_fake = criterion(d_fake, fake_labels)
            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            opt_d.step()

            # --- Generator ---
            opt_g.zero_grad()
            d_fake_for_g = disc(fake)
            loss_g = criterion(d_fake_for_g, torch.full((bs,), 0.9, device=device_t))
            loss_g.backward()
            opt_g.step()

            g_losses.append(loss_g.item())
            d_losses.append(loss_d.item())

        mean_g, mean_d = sum(g_losses) / len(g_losses), sum(d_losses) / len(d_losses)
        history["g_loss"].append(mean_g)
        history["d_loss"].append(mean_d)
        if epoch == 1 or epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            print(f"epoch {epoch:4d}/{epochs}  g_loss={mean_g:.3f}  d_loss={mean_d:.3f}")

    torch.save({"generator_state": gen.state_dict(), "image_size": image_size,
                "latent_dim": LATENT_DIM}, out / "generator.pt")
    with torch.no_grad():
        gen.eval()
        samples = gen(fixed_noise).cpu()
    grid = make_grid(samples, nrow=4, normalize=True, value_range=(-1, 1))
    _save_tensor_image(grid, out / "sample_grid.png")

    return history


def _save_tensor_image(t: torch.Tensor, path: Path) -> None:
    arr = (t.permute(1, 2, 0).numpy() * 255).astype("uint8")
    Image.fromarray(arr).save(path)


def generate_synthetic_images(
    generator_ckpt: str | Path, n: int, out_dir: str | Path,
    class_name: str, source_key: str = "gan", device: str = "cpu",
) -> list[Path]:
    """Generate `n` synthetic images and write them with a self-identifying
    filename prefix, never indistinguishable from real ingest output."""
    device_t = torch.device(device)
    ckpt = torch.load(generator_ckpt, map_location=device_t, weights_only=False)
    gen = Generator(ckpt["latent_dim"], ckpt["image_size"]).to(device_t)
    gen.load_state_dict(ckpt["generator_state"])
    gen.eval()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    with torch.no_grad():
        for i in range(n):
            z = torch.randn(1, ckpt["latent_dim"], device=device_t)
            img_t = gen(z)[0].cpu()
            arr = ((img_t.permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255).clip(0, 255).astype("uint8")
            p = out / f"{source_key}__{class_name}__synthetic_{i:04d}.png"
            Image.fromarray(arr).save(p)
            paths.append(p)
    return paths
