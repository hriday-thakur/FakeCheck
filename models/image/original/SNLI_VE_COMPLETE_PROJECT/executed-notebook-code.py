# CELL 1
%pip -q install "transformers==4.49.0" "huggingface_hub>=0.26,<1" safetensors

# CELL 2
import os, json, hashlib, shutil, gc, random, time, math
from pathlib import Path
os.environ.update(HF_HUB_DISABLE_PROGRESS_BARS="1", TOKENIZERS_PARALLELISM="false", HF_HOME="/tmp/p07_hf")
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from huggingface_hub import snapshot_download, HfApi
import pyarrow.parquet as pq
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_auc_score
from transformers import ViltModel, ViltProcessor, BridgeTowerModel, BridgeTowerProcessor, BlipProcessor, BlipForImageTextRetrieval
from transformers.utils import logging
logging.set_verbosity_error()

SMOKE = False
LOCAL_DATA = None
RESUME = None
PREPARED_DATA = None
SEED = 42
N = 90 if SMOKE else 30000
EVAL_N = 30 if SMOKE else 6000
EPOCHS = 1 if SMOKE else 4
LABELS = ["entailment", "neutral", "contradiction"]
MODELS = {"blip":"Salesforce/blip-itm-base-coco", "vilt":"dandelin/vilt-b32-mlm", "bridgetower":"BridgeTower/bridgetower-base"}
ROOT = Path("/kaggle/working") / f"snlive-three-paired-large-v2-{N}-{EVAL_N}-{SEED}"
if RESUME and not ROOT.exists():
    shutil.copytree(RESUME, ROOT)
ROOT.mkdir(parents=True, exist_ok=True)
if RESUME:
    assert (ROOT/"versions.json").is_file(), "Select a saved three-model run for RESUME"
versions_file=ROOT/"versions.json"
if versions_file.exists():
    versions=json.loads(versions_file.read_text())
else:
    api=HfApi()
    versions={k:api.model_info(v).sha for k,v in MODELS.items()}
    versions["dataset"]=None if LOCAL_DATA else api.dataset_info("sedrickkeh/snli-ve").sha
    versions_file.write_text(json.dumps(versions,indent=2))
assert torch.cuda.device_count() >= 1, "Enable a GPU accelerator before training."
G = min(2, torch.cuda.device_count())
def seed(n):
    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)
    torch.cuda.manual_seed_all(n)
seed(SEED)
print([torch.cuda.get_device_name(i) for i in range(G)], "Free GB", round(shutil.disk_usage(ROOT).free/1e9,1))
def save(obj, path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj,tmp)
    os.replace(tmp,path)
def read(path):
    return torch.load(path, map_location="cpu", weights_only=False)

# CELL 3
old = Path("/kaggle/working") / f"p07-v1-{N}-{SEED}"
data_dir = Path(PREPARED_DATA) if PREPARED_DATA else (old if (old/"pairs.csv").exists() else ROOT)
manifest = data_dir/"pairs.csv"
photos = data_dir/"photos"
if data_dir != ROOT:
    assert manifest.is_file() and photos.is_dir(), "Attach pairs.csv and photos together"
else:
    photos.mkdir(exist_ok=True)
def balanced_eval(f, per_class):
    selected = []
    for _, g in f.groupby("label"):
        assert len(g) >= per_class
        shuffled = g.sample(frac=1, random_state=SEED)
        first = shuffled.drop_duplicates("filename")
        rest = shuffled.loc[~shuffled.index.isin(first.index)]
        selected.append(pd.concat([first, rest]).iloc[:per_class])
    return pd.concat(selected)

if not manifest.exists():
    source = Path(LOCAL_DATA or snapshot_download("sedrickkeh/snli-ve", repo_type="dataset", revision=versions["dataset"], allow_patterns=["data/*.parquet"]))
    parts = []
    for split in ("train","validation","test"):
        files = sorted(source.rglob(split+"*.parquet"))
        if not files and split=="validation":
            files=sorted(source.rglob("dev*.parquet")) or sorted(source.rglob("val*.parquet"))
        assert files, f"Missing {split} Parquet shards under {source}"
        frames = []
        for k,file in enumerate(files):
            f = pq.read_table(file,columns=["filename","hypothesis","label"]).to_pandas()
            f["row"] = np.arange(len(f))
            f["shard"] = k
            frames.append(f)
        f = pd.concat(frames,ignore_index=True)
        assert set(f.label.unique()) == {0,1,2} and f.hypothesis.notna().all()
        if split=="train":
            f = pd.concat([g.sample(N//3,random_state=SEED) for _,g in f.groupby("label")])
        else:
            f = balanced_eval(f, EVAL_N//3)
        f = f.sort_values(["shard","row"]).copy()
        f["split"] = split
        unique = f.drop_duplicates("filename")
        for k,group in unique.groupby("shard"):
            wanted = dict(zip(group.row,group.filename))
            offset = 0
            for batch in pq.ParquetFile(files[k]).iter_batches(batch_size=128,columns=["image"]):
                for i in range(len(batch)):
                    name = wanted.pop(offset+i,None)
                    if name is None:
                        continue
                    assert name == Path(name).name
                    dest = photos/name
                    if not dest.exists():
                        data = batch.column(0)[i].as_py()["bytes"]
                        tmp = dest.with_suffix(".part")
                        tmp.write_bytes(data)
                        tmp.replace(dest)
                offset += len(batch)
                if not wanted:
                    break
            assert not wanted
        parts.append(f)
    pairs = pd.concat(parts,ignore_index=True)
    assert pairs.groupby("filename").split.nunique().max()==1, "Official image groups overlap"
    pairs.to_csv(ROOT/"pairs.tmp",index=False)
    (ROOT/"pairs.tmp").replace(manifest)
pairs = pd.read_csv(manifest,keep_default_na=False)
# Prepared data may contain the previous full evaluation splits. Never modify it in place.
if data_dir != ROOT:
    parts = [pairs[pairs.split=="train"]]
    for split in ("validation", "test"):
        f = pairs[pairs.split==split]
        parts.append(balanced_eval(f, EVAL_N//3).sort_index() if len(f)!=EVAL_N else f)
    pairs = pd.concat(parts, ignore_index=True)
    manifest = ROOT/"pairs.csv"
    pairs.to_csv(manifest, index=False)
for split, count in (("train", N), ("validation", EVAL_N), ("test", EVAL_N)):
    counts = pairs[pairs.split==split].label.value_counts().to_dict()
    assert counts == {0:count//3, 1:count//3, 2:count//3}, (split, counts)
assert pairs.groupby("filename").split.nunique().max()==1, "Official image groups overlap"
display(pd.crosstab(pairs.split,pairs.label).rename(columns=dict(enumerate(LABELS))).join(pairs.groupby("split").filename.nunique().rename("unique_images")))
print("Sample SHA256", hashlib.sha256(manifest.read_bytes()).hexdigest())
digest=hashlib.sha256(manifest.read_bytes()).hexdigest()
stamp=ROOT/"sample.sha256"
if stamp.exists():
    assert stamp.read_text()==digest, "Checkpoints belong to a different sample"
stamp.write_text(digest)
assert all((photos/name).is_file() for name in pairs.filename.unique())

# CELL 4
class Rows(Dataset):
    def __init__(self, frame, image):
        self.frame = frame.reset_index(drop=True)
        self.image = image
    def __len__(self):
        return len(self.frame)
    def __getitem__(self,i):
        r = self.frame.iloc[i]
        pic = None
        if self.image:
            with Image.open(photos/r.filename) as im:
                pic = im.convert("RGB")
        return r.hypothesis, pic, int(r.label)

def processor(kind):
    cls = {"vilt":ViltProcessor,"bridgetower":BridgeTowerProcessor,"blip":BlipProcessor}[kind]
    return cls.from_pretrained(MODELS[kind],revision=versions[kind])

def collator(kind,proc):
    def pack(rows):
        texts, pics, labels = map(list,zip(*rows))
        inputs=proc(text=texts,images=pics,padding=True,truncation=True,
                    max_length=40 if kind=="vilt" else 64,return_tensors="pt")
        return dict(inputs),torch.tensor(labels)
    return pack

class Branch(nn.Module):
    def __init__(self,kind):
        super().__init__()
        self.kind = kind
        cls = {"blip":BlipForImageTextRetrieval,"vilt":ViltModel,"bridgetower":BridgeTowerModel}[kind]
        self.base=cls.from_pretrained(MODELS[kind],revision=versions[kind])
        self.base.requires_grad_(False)
        width=768
        if kind=="vilt":
            blocks=list(self.base.encoder.layer[-4:])+[self.base.layernorm,self.base.pooler]
        elif kind=="bridgetower":
            blocks=list(self.base.cross_modal_text_layers[-2:])+list(self.base.cross_modal_image_layers[-2:])
            blocks.extend([self.base.cross_modal_text_pooler,self.base.cross_modal_image_pooler])
            width=2*self.base.config.hidden_size
        else:
            blocks=list(self.base.text_encoder.encoder.layer[-4:])
        for block in blocks:
            block.requires_grad_(True)
        self.project=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,256),nn.GELU(),nn.LayerNorm(256))
        self.head=nn.Sequential(nn.Dropout(.15),nn.Linear(256,3))
    def forward(self,**x):
        if self.kind in ("vilt","bridgetower"):
            raw=self.base(**x).pooler_output
        else:
            vision = self.base.vision_model(pixel_values=x["pixel_values"]).last_hidden_state
            mask = torch.ones(vision.shape[:2],device=vision.device,dtype=torch.long)
            raw = self.base.text_encoder(input_ids=x["input_ids"],attention_mask=x["attention_mask"],
                encoder_hidden_states=vision,encoder_attention_mask=mask,return_dict=True).last_hidden_state[:,0]
        z = self.project(raw)
        return self.head(z),z

class Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.identity = nn.Parameter(torch.randn(3,256)*.02)
        self.attn = nn.MultiheadAttention(256,4,dropout=.1,batch_first=True)
        self.norm = nn.LayerNorm(256)
        self.head = nn.Sequential(nn.Flatten(),nn.Linear(768,256),nn.GELU(),nn.Dropout(.2),nn.Linear(256,3))
    def forward(self,features):
        tokens = features + self.identity
        mask = torch.eye(3,dtype=torch.bool,device=tokens.device)
        out,weights = self.attn(tokens,tokens,tokens,attn_mask=mask)
        return self.head(self.norm(out+tokens)),weights

def branch_path(k):
    return ROOT/(k+"-best.pt")
def restore(k,net):
    state=read(branch_path(k))
    missing,extra=net.load_state_dict(state,strict=False)
    assert not extra and all(n not in missing for n,p in net.named_parameters() if p.requires_grad)
def trainable(net):
    return {n:p.detach().cpu() for n,p in net.named_parameters() if p.requires_grad}
def parallel(net):
    net=net.cuda()
    return nn.DataParallel(net,device_ids=list(range(G))) if G>1 else net

class Cached(Dataset):
    def __init__(self,split):
        self.arrays=[np.load(ROOT/f"{k}-{split}.npy",mmap_mode="r") for k in MODELS]
        self.labels=pairs.loc[pairs.split==split,"label"].to_numpy()
        assert all(len(x)==len(self.labels) for x in self.arrays)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self,i):
        return {"features":torch.from_numpy(np.stack([a[i] for a in self.arrays]).astype("float32"))},int(self.labels[i])

def loader(k,split,proc,batch,shuffle=False):
    ds = Cached(split) if k=="fusion" else Rows(pairs[pairs.split==split],True)
    return DataLoader(ds,batch_size=batch,shuffle=shuffle,num_workers=2 if k!="fusion" else 0,
        pin_memory=True,collate_fn=None if k=="fusion" else collator(k,proc))

def move(x):
    return {k:v.cuda(non_blocking=True) for k,v in x.items()}

# CELL 5
def atomic(obj,path):
    tmp=path.with_suffix(".tmp")
    torch.save(obj,tmp)
    tmp.replace(path)

@torch.inference_mode()
def evaluate(model,batches):
    model.eval()
    probs,targets=[],[]
    for x,y in batches:
        with torch.autocast("cuda",dtype=torch.float16):
            logits,_=model(**move(x))
        probs.append(logits.float().softmax(-1).cpu().numpy())
        targets.extend(y.tolist())
    return np.array(targets),np.concatenate(probs)

def view_history(k,h):
    clear_output(wait=True)
    df=pd.DataFrame(h)
    display(df.tail(4))
    fig,ax=plt.subplots(1,2,figsize=(9,3))
    df.plot(x="epoch",y=["train_loss","val_loss"],ax=ax[0])
    df.plot(x="epoch",y=["train_acc","val_acc","macro_f1"],ax=ax[1],ylim=(0,1))
    fig.suptitle(k)
    fig.tight_layout()
    fig.savefig(ROOT/f"{k}-curves.png")
    plt.show()
    plt.close(fig)

def train(k,batch):
    seed(SEED)
    net=Fusion() if k=="fusion" else Branch(k)
    proc=None if k=="fusion" else processor(k)
    model=parallel(net)
    groups=[{"params":[p for n,p in net.named_parameters() if p.requires_grad and n.startswith("base.")],"lr":2e-5},
            {"params":[p for n,p in net.named_parameters() if p.requires_grad and not n.startswith("base.")],"lr":3e-4}]
    opt=torch.optim.AdamW(groups,weight_decay=.01)
    scaler=torch.amp.GradScaler("cuda")
    last=ROOT/f"{k}-latest.pt"
    history=[]
    if last.exists():
        s=read(last)
        net.load_state_dict(s["weights"],strict=False)
        opt.load_state_dict(s["optimizer"])
        scaler.load_state_dict(s["scaler"])
        history=s["history"]
        del s
    goal=128 if k=="fusion" else 32
    accum=max(1,goal//batch)
    for epoch in range(len(history)+1,EPOCHS+1):
        seed(SEED+epoch)
        model.train()
        if k=="blip":
            net.base.vision_model.eval()
        batches=loader(k,"train",proc,batch,True)
        size=len(batches.dataset)
        total=correct=seen=0
        print(f'{k}: global batch {batch}, accumulation {accum}')
        opt.zero_grad(set_to_none=True)
        start=time.monotonic()
        for i,(x,y) in enumerate(batches):
            y=y.cuda(non_blocking=True)
            window=(i//accum)*accum
            denom=min(accum*batch,size-window*batch)
            progress=((epoch-1)+(i+1)/len(batches))/EPOCHS
            factor=min(progress/.06,1)*(.1+.9*(1+math.cos(math.pi*progress))/2)
            for group,lr in zip(opt.param_groups,(2e-5,3e-4)):
                group["lr"]=lr*factor
            with torch.autocast("cuda",dtype=torch.float16):
                logits,_=model(**move(x))
                loss=nn.functional.cross_entropy(logits,y,reduction="sum")
            scaler.scale(loss/denom).backward()
            if (i+1)%accum==0 or i+1==len(batches):
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(net.parameters(),1.)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            total+=loss.item()
            correct+=(logits.argmax(1)==y).sum().item()
            seen+=len(y)
            if i%200==0:
                print(f"{k} epoch {epoch} | {seen}/{size} pairs | loss {total/seen:.3f}",end="\r",flush=True)
        train_seconds=time.monotonic()-start
        y,p=evaluate(model,loader(k,"validation",proc,batch))
        pred=p.argmax(1)
        row=dict(epoch=epoch,global_batch=batch,train_pairs_per_second=round(seen/train_seconds,1),train_loss=total/seen,train_acc=correct/seen,
                 val_loss=float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean()),
                 val_acc=accuracy_score(y,pred),macro_f1=f1_score(y,pred,average="macro"),seconds=round(time.monotonic()-start))
        score=lambda r:(r["val_acc"],r["macro_f1"])
        if not history or score(row)>max(map(score,history)):
            atomic(trainable(net),branch_path(k))
        history.append(row)
        atomic(dict(weights=trainable(net),optimizer=opt.state_dict(),scaler=scaler.state_dict(),history=history),last)
        pd.DataFrame(history).to_csv(ROOT/f"{k}-history.csv",index=False)
        view_history(k,history)
    pd.DataFrame(history).to_csv(ROOT/f"{k}-history.csv",index=False)

def read(path):
    return torch.load(path,map_location="cpu",weights_only=False)

def fit(k):
    batch={"blip":8,"vilt":16,"bridgetower":4,"fusion":64}[k]*G
    while True:
        try:
            train(k,batch)
            return
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or batch<=G:
                raise
        gc.collect()
        torch.cuda.empty_cache()
        batch=max(G,batch//2)
        print("Retrying from completed epoch with global batch",batch)

# CELL 6
@torch.inference_mode()
def cache(k,split):
    target=ROOT/f"{k}-{split}.npy"
    if target.exists():
        return
    net=Branch(k)
    restore(k,net)
    model=parallel(net).eval()
    proc=processor(k)
    tmp=target.with_suffix(".partial")
    size=(pairs.split==split).sum()
    arr=np.lib.format.open_memmap(tmp,mode="w+",dtype="float16",shape=(size,256))
    offset=0
    for x,_ in loader(k,split,proc,2*G):
        with torch.autocast("cuda",dtype=torch.float16):
            _,z=model(**move(x))
        end=offset+len(z)
        arr[offset:end]=z.cpu().half().numpy()
        offset=end
    assert offset==size
    arr.flush()
    del arr
    tmp.replace(target)

def move(x):
    return {k:v.cuda(non_blocking=True) for k,v in x.items()}

def preflight(kind):
    net=Branch(kind).cuda().train()
    available=Rows(pairs[pairs.split=="train"].iloc[:2],True)
    x,y=collator(kind,processor(kind))([available[0],available[1]])
    with torch.autocast("cuda",dtype=torch.float16):
        logits,embedding=net(**move(x))
        loss=nn.functional.cross_entropy(logits,y.cuda())
    assert embedding.shape==(2,256) and torch.isfinite(loss)
    loss.backward()
    print(kind,"forward and backward passed")

for kind in MODELS:
    preflight(kind)
    gc.collect()
    torch.cuda.empty_cache()

for kind in MODELS:
    fit(kind)
    gc.collect()
    torch.cuda.empty_cache()
    for split in ("train","validation"):
        cache(kind,split)
    print(kind,"ready")

# CELL 7
fit("fusion")
runs=list(MODELS)+["fusion"]
def rank(k):
    h=pd.read_csv(ROOT/f"{k}-history.csv")
    return h.sort_values(["val_acc","macro_f1"],ascending=False).iloc[0]
table=pd.DataFrame({k:rank(k) for k in runs}).T.sort_values(["val_acc","macro_f1"],ascending=False)
winner=table.index[0]
table.to_csv(ROOT/"validation-comparison.csv")
(ROOT/"selection.json").write_text(json.dumps(dict(model=winner,epoch=int(table.iloc[0].epoch),train_pairs=N)))
display(table)
table[["val_acc","macro_f1"]].plot.bar(ylim=(0,1),rot=0,title="Validation comparison on identical pairs")
plt.tight_layout()
plt.savefig(ROOT/"validation-comparison.png")
plt.show()
print("Selected before test evaluation:",winner)

# CELL 8
import ast
import re
import struct
import numpy as np

for k in MODELS:
    for split in ("train", "validation", "test"):
        path = ROOT / f"{k}-{split}.npy"

        if not path.exists():
            continue

        try:
            np.load(path, mmap_mode="r")
            print(f"{path.name}: Already valid")
            continue
        except ValueError:
            pass

        with open(path, "rb") as f:
            version = np.lib.format.read_magic(f)
            fmt = "<H" if version == (1, 0) else "<I"
            length = struct.unpack(
                fmt, f.read(struct.calcsize(fmt))
            )[0]

            header = f.read(length).decode(
                "utf8" if version == (3, 0) else "latin1"
            )
            offset = f.tell()

        fixed = re.sub(
            r"(?:np|numpy)\.int(?:32|64)\((\d+)\)",
            r"\1",
            header
        )

        meta = ast.literal_eval(fixed)
        shape = tuple(map(int, meta["shape"]))
        dtype = np.dtype(meta["descr"])

        expected = (int((pairs.split == split).sum()), 256)

        assert shape == expected, (shape, expected)
        assert dtype == np.dtype("float16")
        assert path.stat().st_size == (
            offset + shape[0] * shape[1] * dtype.itemsize
        )

        data = np.memmap(
            path, dtype=dtype, mode="r",
            offset=offset, shape=shape,
            order="F" if meta["fortran_order"] else "C"
        )

        assert np.isfinite(data).all()

        temp = path.with_suffix(".fixed.npy")
        np.save(temp, np.asarray(data))
        del data

        np.load(temp, mmap_mode="r")
        temp.replace(path)

        print(f"{path.name}: FIXED {shape}")

print("\nCache repair completed.")

# CELL 9
@torch.inference_mode()
def cache(k,split):
    target=ROOT/f"{k}-{split}.npy"
    if target.exists():
        return
    net=Branch(k)
    restore(k,net)
    model=parallel(net).eval()
    proc=processor(k)
    tmp=target.with_suffix(".partial")
    size=int((pairs.split==split).sum())
    arr=np.lib.format.open_memmap(tmp,mode="w+",dtype="float16",shape=(size,256))
    offset=0
    for x,_ in loader(k,split,proc,2*G):
        with torch.autocast("cuda",dtype=torch.float16):
            _,z=model(**move(x))
        end=offset+len(z)
        arr[offset:end]=z.cpu().half().numpy()
        offset=end
    assert offset==size
    arr.flush()
    del arr
    tmp.replace(target)

def move(x):
    return {k:v.cuda(non_blocking=True) for k,v in x.items()}

def preflight(kind):
    net=Branch(kind).cuda().train()
    available=Rows(pairs[pairs.split=="train"].iloc[:2],True)
    x,y=collator(kind,processor(kind))([available[0],available[1]])
    with torch.autocast("cuda",dtype=torch.float16):
        logits,embedding=net(**move(x))
        loss=nn.functional.cross_entropy(logits,y.cuda())
    assert embedding.shape==(2,256) and torch.isfinite(loss)
    loss.backward()
    print(kind,"forward and backward passed")

for kind in MODELS:
    preflight(kind)
    gc.collect()
    torch.cuda.empty_cache()

for kind in MODELS:
    fit(kind)
    gc.collect()
    torch.cuda.empty_cache()
    for split in ("train","validation"):
        cache(kind,split)
    print(kind,"ready")

# CELL 10
fit("fusion")
runs=list(MODELS)+["fusion"]
def rank(k):
    h=pd.read_csv(ROOT/f"{k}-history.csv")
    return h.sort_values(["val_acc","macro_f1"],ascending=False).iloc[0]
table=pd.DataFrame({k:rank(k) for k in runs}).T.sort_values(["val_acc","macro_f1"],ascending=False)
winner=table.index[0]
table.to_csv(ROOT/"validation-comparison.csv")
(ROOT/"selection.json").write_text(json.dumps(dict(model=winner,epoch=int(table.iloc[0].epoch),train_pairs=N)))
display(table)
table[["val_acc","macro_f1"]].plot.bar(ylim=(0,1),rot=0,title="Validation comparison on identical pairs")
plt.tight_layout()
plt.savefig(ROOT/"validation-comparison.png")
plt.show()
print("Selected before test evaluation:",winner)

# CELL 11
if winner=="fusion":
    for k in MODELS:
        cache(k,"test")
net=Fusion() if winner=="fusion" else Branch(winner)
restore(winner,net)
model=parallel(net)
proc=None if winner=="fusion" else processor(winner)
y,p=evaluate(model,loader(winner,"test",proc,2*G if winner!="fusion" else 128*G))
del model,net
gc.collect()
torch.cuda.empty_cache()
pred=p.argmax(1)
report=pd.DataFrame(classification_report(y,pred,labels=[0,1,2],target_names=LABELS,output_dict=True,zero_division=0)).T
report.to_csv(ROOT/"test-class-report.csv")
display(report)
print("Test accuracy",accuracy_score(y,pred),"Macro ROC-AUC",roc_auc_score(y,p,multi_class="ovr",average="macro"))
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay(confusion_matrix(y,pred,labels=[0,1,2]),display_labels=LABELS).plot(cmap="Blues")
plt.tight_layout()
plt.savefig(ROOT/"test-confusion.png")
plt.show()
test=pairs[pairs.split=="test"].reset_index(drop=True).copy()
test["prediction"] = pred
test["confidence"] = p.max(1)
for i,name in enumerate(LABELS):
    test["p_"+name]=p[:,i]
test.to_csv(ROOT/"test-predictions.csv",index=False)
errors=test[test.label!=test.prediction].nlargest(15,"confidence")
display(errors[["filename","hypothesis","label","prediction","confidence"]])
def show_cases(rows):
    import textwrap
    if rows.empty:
        return
    fig,axes=plt.subplots(math.ceil(len(rows)/3),3,figsize=(12,3.9*math.ceil(len(rows)/3)),squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax,r in zip(axes.flat,rows.itertuples()):
        with Image.open(photos/r.filename) as im:
            ax.imshow(im)
        ax.set_title(textwrap.fill(r.hypothesis,39)+"\n"+LABELS[r.label]+" → "+LABELS[r.prediction],fontsize=9)
    fig.tight_layout()
    fig.savefig(ROOT/"difficult-cases.png")
    plt.show()
show_cases(errors)

# CELL 12
@torch.inference_mode()
def predict_rows(rows,k):
    branches=list(MODELS) if k=="fusion" else [k]
    reps=[]
    for b in branches:
        net=Branch(b)
        restore(b,net)
        net=net.cuda().eval()
        pack=collator(b,processor(b))
        logits,features=[],[]
        for i in range(0,len(rows),4):
            x,_=pack(rows[i:i+4])
            with torch.autocast("cuda",dtype=torch.float16):
                out,z=net(**move(x))
            logits.append(out.float().cpu())
            features.append(z.float().cpu())
        reps.append(torch.cat(features))
        del net
    if k=="fusion":
        net=Fusion()
        restore(k,net)
        out,_=net.eval()(torch.stack(reps,dim=1))
    else:
        out=torch.cat(logits)
    return out.softmax(-1).numpy()

sample=test.sample(n=min(96,len(test)),random_state=SEED)
rows=list(Rows(sample,True))
blank=lambda im:Image.new("RGB",im.size,(127,127,127))
swaps=[]
for i,(txt,im,label) in enumerate(rows):
    j=next(j for j in range(len(rows)) if sample.iloc[j].filename!=sample.iloc[i].filename)
    swaps.append((txt,rows[j][1],label))
conditions={"complete":rows,"image missing":[(t,blank(im),y) for t,im,y in rows],
            "text missing":[("",im,y) for t,im,y in rows],"swapped photograph":swaps}
# Evaluate all perturbations together to load each backbone just once.
txt,im,label=rows[0]
words=txt.split()[:20]
edits=[]
names=[]
for i in range(4):
    copy=im.copy()
    w,h=im.size
    x=(i%2)*w//2
    top=(i//2)*h//2
    copy.paste((127,127,127),(x,top,(i%2+1)*w//2,(i//2+1)*h//2))
    edits.append((txt,copy,label))
    names.append(["top left","top right","bottom left","bottom right"][i])
for i,word in enumerate(words):
    all_words=txt.split()
    edits.append((" ".join(all_words[:i]+all_words[i+1:]),im,label))
    names.append(f"{i+1}. {word}")
all_probs=predict_rows([r for group in conditions.values() for r in group]+edits,winner)
m=len(rows)
base=all_probs[:m]
target=np.array([r[2] for r in rows])
metrics=[]
for i,name in enumerate(conditions):
    q=all_probs[i*m:(i+1)*m]
    metrics.append(dict(condition=name,accuracy=accuracy_score(target,q.argmax(1)) if i<3 else np.nan,
                        prediction_changed=(q.argmax(1)!=base.argmax(1)).mean(),
                        original_class_prob_drop=(base.max(1)-q[np.arange(m),base.argmax(1)]).mean()))
diagnostics=pd.DataFrame(metrics)
display(diagnostics)
diagnostics.to_csv(ROOT/"modality-checks.csv",index=False)
c=base[0].argmax()
drops=base[0,c]-all_probs[4*m:,c]
fig,(ax,bx)=plt.subplots(1,2,figsize=(12,max(4,len(names)*.27)))
ax.imshow(im)
ax.axis("off")
import textwrap
ax.set_title(textwrap.fill(txt,40)+"\nPrediction: "+LABELS[c])
bx.barh(range(len(names)),drops)
bx.set_yticks(range(len(names)),names)
bx.invert_yaxis()
bx.set_xlabel("Probability drop after occlusion")
fig.tight_layout()
fig.savefig(ROOT/"occlusion.png")
plt.show()

# CELL 13
import os
import json
import shutil
import zipfile
import platform
import subprocess
import torch
import numpy as np
import pandas as pd

from pathlib import Path
from IPython.display import FileLink, display

# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(ROOT)
OUTPUT = Path("/kaggle/working")
ZIP_PATH = OUTPUT / "SNLI_VE_COMPLETE_PROJECT.zip"

assert ROOT.exists(), f"Results directory not found: {ROOT}"

# ============================================================
# 2. SAVE COMPLETE EXPERIMENT INFORMATION
# ============================================================

metadata = {}

variables = [
    "MODELS", "SEED", "EPOCHS", "LABELS",
    "N", "winner", "versions", "BATCH",
    "BATCHES", "LR", "LRS", "CONFIG",
    "DEVICE", "AMP", "PATIENCE",
    "GRAD_ACCUM", "TARGET_BATCH"
]

for name in variables:
    if name in globals():
        metadata[name] = globals()[name]

metadata["environment"] = {
    "python": platform.python_version(),
    "pytorch": torch.__version__,
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "cuda": torch.version.cuda,
    "gpu_count": torch.cuda.device_count(),
    "gpu_names": [
        torch.cuda.get_device_name(i)
        for i in range(torch.cuda.device_count())
    ]
}

(ROOT / "complete-experiment-info.json").write_text(
    json.dumps(metadata, indent=4, default=str)
)

# ============================================================
# 3. SAVE DATASET INFORMATION
# ============================================================

if "pairs" in globals():

    pairs.to_csv(
        ROOT / "complete-dataset-pairs.csv",
        index=False
    )

    pairs.groupby(["split", "label"]).size().unstack(
        fill_value=0
    ).to_csv(ROOT / "dataset-class-distribution.csv")

    pairs.groupby("split").size().to_csv(
        ROOT / "dataset-split-distribution.csv"
    )

# ============================================================
# 4. SAVE SOFTWARE DEPENDENCIES
# ============================================================

try:
    requirements = subprocess.check_output(
        ["python", "-m", "pip", "freeze"],
        text=True
    )

    (ROOT / "requirements.txt").write_text(requirements)

except Exception as e:
    print("Dependency export warning:", e)

# ============================================================
# 5. SAVE EXECUTED NOTEBOOK CODE
# ============================================================

try:
    history = get_ipython().history_manager.input_hist_raw

    (ROOT / "executed-notebook-code.py").write_text(
        "\n\n".join(
            f"# CELL {i}\n{code}"
            for i, code in enumerate(history)
            if code.strip()
        )
    )

except Exception as e:
    print("Code history export warning:", e)

# ============================================================
# 6. FIND ALL EXPERIMENT FILES
# ============================================================

files = [
    f for f in ROOT.rglob("*")
    if f.is_file()
    and f.resolve() != ZIP_PATH.resolve()
]

total_bytes = sum(f.stat().st_size for f in files)

print("=" * 70)
print("COMPLETE PROJECT EXPORT")
print("=" * 70)

print(f"Source directory: {ROOT}")
print(f"Total files: {len(files)}")
print(f"Uncompressed size: {total_bytes / 1024**3:.3f} GB")

# ============================================================
# 7. VERIFY MODEL CHECKPOINTS
# ============================================================

print("\nMODEL CHECKPOINTS:")

for model in ["blip", "vilt", "bridgetower", "fusion"]:

    checkpoints = [
        f for f in files
        if model in f.name.lower()
        and f.suffix.lower() in {
            ".pt", ".pth", ".ckpt",
            ".safetensors", ".bin"
        }
    ]

    if checkpoints:
        for f in checkpoints:
            print(
                f"FOUND: {f.name} | "
                f"{f.stat().st_size / 1024**2:.2f} MB"
            )
    else:
        print(f"WARNING: {model} checkpoint not found")

# ============================================================
# 8. CREATE FILE MANIFEST
# ============================================================

manifest = pd.DataFrame([
    {
        "file": str(f.relative_to(ROOT)),
        "extension": f.suffix,
        "size_bytes": f.stat().st_size,
        "size_mb": round(
            f.stat().st_size / 1024**2, 4
        )
    }
    for f in files
])

manifest.to_csv(
    ROOT / "complete-file-manifest.csv",
    index=False
)

# Refresh file list to include manifest
files = sorted(
    f for f in ROOT.rglob("*")
    if f.is_file()
    and f.resolve() != ZIP_PATH.resolve()
)

# ============================================================
# 9. CREATE COMPLETE ZIP — NO FILE TYPE EXCLUSIONS
# ============================================================

if ZIP_PATH.exists():
    ZIP_PATH.unlink()

with zipfile.ZipFile(
    ZIP_PATH,
    mode="w",
    compression=zipfile.ZIP_DEFLATED,
    compresslevel=1,
    allowZip64=True
) as archive:

    for i, file in enumerate(files, start=1):

        archive.write(
            file,
            arcname=str(file.relative_to(ROOT))
        )

        print(
            f"[{i}/{len(files)}] "
            f"{file.relative_to(ROOT)} | "
            f"{file.stat().st_size / 1024**2:.2f} MB"
        )

# ============================================================
# 10. VERIFY ZIP INTEGRITY
# ============================================================

with zipfile.ZipFile(ZIP_PATH, "r") as archive:

    corrupted = archive.testzip()

    if corrupted is not None:
        raise RuntimeError(
            f"Corrupted ZIP entry: {corrupted}"
        )

    archived_files = archive.namelist()

    print("\nZIP integrity check: PASSED")
    print(f"Archived files: {len(archived_files)}")

# ============================================================
# 11. FINAL DOWNLOAD
# ============================================================

print("\n" + "=" * 70)
print("COMPLETE EXPORT SUCCESSFUL")
print("=" * 70)

print(f"ZIP: {ZIP_PATH.name}")
print(f"Size: {ZIP_PATH.stat().st_size / 1024**3:.3f} GB")
print(f"Location: {ZIP_PATH}")

os.chdir("/kaggle/working")

display(FileLink(ZIP_PATH.name))