"""Download the subsets of public datasets used in this project into ../data.

  NYU Depth v2 (vikhyatk/nyu_depth_v2): in-air RGB + metric depth -> synthetic water
      6 train shards spread over the 157 (different rooms) + the official 654-image test split
  UIEB (Hikari0608/UIEB): 890 real underwater images with reference "enhanced" images
  Sea-thru (bhowmikabhimanyu/seathru-dataset): real linear RAW images with SfM metric depth,
      colour charts in scene. Only the chart frames used in Akkaynak & Treibitz 2019 Fig. 6
      plus a few evenly spaced frames per dive site.
"""
import argparse
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1] / "data"


def get(repo, filename, sub):
    out = ROOT / sub
    out.mkdir(parents=True, exist_ok=True)
    p = hf_hub_download(repo, filename, repo_type="dataset", local_dir=out)
    print("ok", p, flush=True)


def nyu():
    files = [f.rfilename for f in HfApi().dataset_info("vikhyatk/nyu_depth_v2").siblings]
    train = sorted(f for f in files if "/train-" in f)
    val = sorted(f for f in files if "/validation-" in f)
    for f in [train[i] for i in (0, 26, 52, 78, 104, 130)] + val:
        get("vikhyatk/nyu_depth_v2", f, "nyu")


def uieb():
    files = [f.rfilename for f in HfApi().dataset_info("Hikari0608/UIEB").siblings]
    for f in files:
        if f.endswith(".parquet"):
            get("Hikari0608/UIEB", f, "uieb")


def seathru(per_site=4):
    names = [f.rfilename for f in HfApi().dataset_info("bhowmikabhimanyu/seathru-dataset").siblings]
    pngs = sorted(n for n in names if "/linearPNG/" in n)
    # chart frames shown in the Sea-thru paper
    wanted = {"D1/D1/linearPNG/T_S03272.png", "D2/D2/linearPNG/T_S03647.png",
              "D3/D3/linearPNG/T_S04910.png", "D4/D4/linearPNG/RGT_0209.png",
              "D5/D5/linearPNG/LFT_3374.png"}
    for site in ["D1", "D2", "D3", "D4", "D5"]:
        s = [p for p in pngs if p.startswith(site + "/")]
        wanted.update(s[i] for i in range(0, len(s), max(1, len(s) // per_site))[:per_site])
    for png in sorted(wanted):
        stem = png.split("/")[-1][:-4]
        depth = [n for n in names if n.split("/")[0] == png.split("/")[0]
                 and n.endswith(f"depth{stem}.tif")]
        get("bhowmikabhimanyu/seathru-dataset", png, "seathru")
        for d in depth:
            get("bhowmikabhimanyu/seathru-dataset", d, "seathru")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="+", choices=["nyu", "uieb", "seathru"])
    for w in ap.parse_args().which:
        globals()[w]()
