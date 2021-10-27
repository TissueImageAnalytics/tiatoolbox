# %%
import copy
import json
import os
import pathlib
import random
import shutil
import sys
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append("/home/dang/storage_1/workspace/tiatoolbox")

# ! save_yaml, save_as_json => need same name, need to factor out jsonify
from tiatoolbox.utils.misc import save_as_json

# %%


def load_json(path):
    """Helper to load json file."""
    with open(path, "r") as fptr:
        json_dict = json.load(fptr)
    return json_dict


def rmdir(dir_path):
    """Helper function to remove directory."""
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    return


def rm_n_mkdir(dir_path):
    """Helper function to remove then re-create directory."""
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)
    return


def recur_find_ext(root_dir, exts):
    """Helper function to recursively get files with extensions.

    Recursively find all files in directories end with the `ext`
    such as `ext=['.png']` . Much faster than glob if the folder
    hierachy is complicated and contain > 1000 files.

    Args:
        root_dir (str): Root directory for searching.
        exts (list): List of extensions to match.

    Returns:
        List of full paths with matched extension in sorted order.

    """
    assert isinstance(exts, list)
    file_path_list = []
    for cur_path, dir_list, file_list in os.walk(root_dir):
        for file_name in file_list:
            file_ext = pathlib.Path(file_name).suffix
            if file_ext in exts:
                full_path = os.path.join(cur_path, file_name)
                file_path_list.append(full_path)
    file_path_list.sort()
    return file_path_list


# %%
SEED = 5
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

ROOT_OUTPUT_DIR = "/home/dang/storage_1/workspace/tiatoolbox/local/code/"
WSI_DIR = "/home/dang/storage_1/dataset/TCGA-LUAD/"
MSK_DIR = None

wsi_paths = recur_find_ext(WSI_DIR, [".svs", ".ndpi"])
wsi_names = [pathlib.Path(v).stem for v in wsi_paths]
msk_paths = None if MSK_DIR is None else [f"{MSK_DIR}/{v}.png" for v in wsi_names]
assert len(wsi_paths) > 0, "No files found."

# %% [markdown]
# ## Generate the data split

# %%
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def generate_split(x, y, train, valid, test):
    """Helper to generate stratified splits."""
    outer_splitter = StratifiedShuffleSplit(
        n_splits=NUM_FOLDS, train_size=train, test_size=valid + test, random_state=SEED
    )
    inner_splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=valid / (valid + test),
        test_size=test / (valid + test),
        random_state=SEED,
    )

    x = np.array(x)
    y = np.array(y)
    split_list = []
    for train_idx, valid_test_idx in outer_splitter.split(x, y):
        train_x = x[train_idx]
        train_y = y[train_idx]

        # holder for valid_test set
        x_ = x[valid_test_idx]
        y_ = y[valid_test_idx]

        # split valid_test into valid and test set
        valid_idx, test_idx = list(inner_splitter.split(x_, y_))[0]
        valid_x = x_[valid_idx]
        valid_y = y_[valid_idx]

        test_x = x_[test_idx]
        test_y = y_[test_idx]
        # integrity check
        assert len(set(train_x).intersection(set(valid_x))) == 0
        assert len(set(valid_x).intersection(set(test_x))) == 0
        assert len(set(train_x).intersection(set(test_x))) == 0

        split_list.append(
            {
                "train": list(zip(train_x, train_y)),
                "valid": list(zip(valid_x, valid_y)),
                "test": list(zip(test_x, test_y)),
            }
        )
    return split_list


# %%

# - debug injection, remove later
wsi_paths = recur_find_ext(
    "/home/dang/storage_1/workspace/tiatoolbox/local/code/data/resnet", [".json"]
)
wsi_names = np.array([pathlib.Path(v).stem for v in wsi_paths])
#

NUM_FOLDS = 5
TEST_RATIO = 0.2
TRAIN_RATIO = 0.8 * 0.9
VALID_RATIO = 0.8 * 0.1
CLINICAL_FILE = (
    "/home/dang/storage_1/workspace/tiatoolbox/local/code/TCGA-BRCA-DX_CLINI.csv"
)
clinical_df = pd.read_csv(CLINICAL_FILE)

patient_uids = clinical_df["PATIENT"].to_numpy()
patient_labels = clinical_df["HER2FinalStatus"].to_numpy()

patient_labels_ = np.full_like(patient_labels, -1)
patient_labels_[patient_labels == "Positive"] = 1
patient_labels_[patient_labels == "Negative"] = 0
sel = patient_labels_ >= 0

patient_labels = patient_uids[sel]
patient_labels = patient_labels_[sel]
clinical_info = OrderedDict(list(zip(patient_uids, patient_labels)))

# retrieve patient code of each WSI, this is basing TCGA bar codes
# https://docs.gdc.cancer.gov/Encyclopedia/pages/TCGA_Barcode/
wsi_patient_codes = np.array(["-".join(v.split("-")[:3]) for v in wsi_names])
wsi_labels = np.array(
    [clinical_info[v] if v in clinical_info else np.nan for v in wsi_patient_codes]
)

# %%

sel = ~np.isnan(wsi_labels)
wsi_labels = wsi_labels[sel]
wsi_names = wsi_names[sel]

label_df = list(zip(wsi_names, wsi_labels))
label_df = pd.DataFrame(label_df, columns=["WSI-CODE", "LABEL"])

x = np.array(label_df["WSI-CODE"].to_list())
y = np.array(label_df["LABEL"].to_list())

# %%
split_list = generate_split(x, y, 0.6, 0.2, 0.2)


# %% [markdown]
# ### Transform the WSI into graph data

# %% [markdown]
# # Generate the stain normalizer for pre-processing patch

# %%
from tiatoolbox.data import stainnorm_target
from tiatoolbox.tools.stainnorm import get_normaliser

target_image = stainnorm_target()
stain_normaliser = get_normaliser("vahadane")
stain_normaliser.fit(target_image)


def stain_norm_func(img):
    return stain_normaliser.transform(img)


# %% [markdown]
# # Define the code for feature extraction


# %%
from tiatoolbox.models import FeatureExtractor, IOSegmentorConfig
from tiatoolbox.models.architecture import CNNExtractor


def extract_deep_feature(save_dir):
    ioconfig = IOSegmentorConfig(
        input_resolutions=[
            {"units": "mpp", "resolution": 0.25},
        ],
        output_resolutions=[
            {"units": "mpp", "resolution": 0.25},
        ],
        patch_input_shape=[512, 512],
        patch_output_shape=[512, 512],
        stride_shape=[512, 512],
        save_resolution={"units": "mpp", "resolution": 8.0},
    )

    model = CNNExtractor("resnet50")
    model.preproc_func = stain_norm_func
    extractor = FeatureExtractor(batch_size=4, model=model)

    rmdir(save_dir)
    output_list = extractor.predict(
        wsi_paths,
        msk_paths,
        mode="wsi",
        ioconfig=ioconfig,
        on_gpu=True,
        crash_on_exception=True,
        save_dir=save_dir,
    )
    return output_list


# %%


def extract_composition_feature(save_dir):
    return []


# %% [markdown]
# Now that we have defined functions for performing WSI feature extraction,
# we now perform the extraction itself. Additionally, we would want to avoid
# un-necessarily re-extracting the WSI features as they are computationally
# expensive in nature. Here, we differentiate these through use case via `CACHE_PATH`
# variable, if `CACHE_PATH = None`, the extraction is performed and the results is save
# into `WSI_FEATURE_DIR`. For ease of organization, we set the
# `WSI_FEATURE_DIR = f'{ROOT_OUTPUT_DIR}/features/'` by default. Otherwise, the paths
# to feature files are queried. We also put an assertion check by then to ensure we have
# the same number of output file as the number of sample cases we loaded above.
# %%
# set to None to extract into WSI_FEATURE_DIR
# else it will load cached data from CACHE_PATH
CACHE_PATH = "/home/dang/storage_1/workspace/tiatoolbox/local/code/data/resnet/"

FEATURE_MODE = "cnn"
WSI_FEATURE_DIR = f"{ROOT_OUTPUT_DIR}/features/"
if CACHE_PATH and os.path.exists(CACHE_PATH):
    # ! check the extension search
    output_list = recur_find_ext(f"{CACHE_PATH}/", [".json"])
elif FEATURE_MODE == "composition":
    output_list = extract_composition_feature(WSI_FEATURE_DIR)
else:
    output_list = extract_deep_feature(WSI_FEATURE_DIR)
# ! put the assertion back later
# assert len(output_list) == len(wsi_names), 'Missing output.'

# %% [markdown]
# With the patches and their features we have just loaded, we construct the graph
# for each WSI using the function provided in our toolbox `tiatoolbox.tools.graph`.
# Like above, if we already have the graph extracted, we will only need to load them back
# via `CACHE_PATH` instead of wasting time on re-doing the job.
#
# ! [@WENQI, a patch is a node here, but are there other cases? We may need to mention here]
# %%

from tiatoolbox.tools.graph import hybrid_clustered_graph


def construct_graph(wsi_name, save_path):
    """Construct graph for one WSI and save to file."""
    positions = np.load(f"{WSI_FEATURE_DIR}/{wsi_name}.position.npy")
    features = np.load(f"{WSI_FEATURE_DIR}/{wsi_name}.features.npy")
    graph_dict = hybrid_clustered_graph(positions[:, :2], features)

    # Write a graph to a JSON file
    with open(save_path, "w") as handle:
        graph_dict = {k: v.tolist() for k, v in graph_dict.items()}
        json.dump(obj=graph_dict, fp=handle)


CACHE_PATH = "/home/dang/storage_1/workspace/tiatoolbox/local/code/data/resnet/"

GRAPH_DIR = f"{ROOT_OUTPUT_DIR}/graph/"
if CACHE_PATH and os.path.exists(CACHE_PATH):
    GRAPH_DIR = CACHE_PATH  # assignment for follow up loading
    graph_paths = recur_find_ext(f"{CACHE_PATH}/", [".json"])
else:
    graph_paths = [construct_graph(v, f"{GRAPH_DIR}/{v}.json") for v in wsi_names]
# ! put the assertion back later
# assert len(graph_paths) == len(wsi_names), 'Missing output.'

# %%
# [markdown]
# ## The dataset

# %%

from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader


class SlideGraphDataset(Dataset):
    def __init__(self, info_list, mode="train", preproc=None):
        self.info_list = info_list
        self.mode = mode
        self.preproc = preproc

    def __getitem__(self, idx):
        info = self.info_list[idx]
        if self.mode != "infer":
            wsi_code, label = info
            # torch.Tensor will create 1-d vector not scalar
            label = torch.tensor(label)
        else:
            wsi_code = info

        with open(f"{GRAPH_DIR}/{wsi_code}.json", "r") as fptr:
            graph_dict = json.load(fptr)
        graph_dict = {k: np.array(v) for k, v in graph_dict.items()}

        if self.preproc is not None:
            graph_dict["x"] = self.preproc(graph_dict["x"])

        graph_dict = {k: torch.tensor(v) for k, v in graph_dict.items()}
        graph = Data(**graph_dict)

        if self.mode != "infer":
            return dict(graph=graph, label=label)
        return dict(graph=graph)

    def __len__(self):
        return len(self.info_list)


# %%
# [markdown]
# ## Entire dataset feature normalization

import joblib
from sklearn.preprocessing import StandardScaler

CACHE_PATH = "/home/dang/storage_1/workspace/tiatoolbox/local/code/data/node_scaler.dat"
SCALER_PATH = (
    "/home/dang/storage_1/workspace/tiatoolbox/local/code/data/node_scaler.dat"
)

if CACHE_PATH and os.path.exists(CACHE_PATH):
    SCALER_PATH = CACHE_PATH  # assignment for follow up loading
    node_scaler = joblib.load(SCALER_PATH)
else:
    wsi_codes = label_df["WSI-CODE"].to_list()
    loader = SlideGraphDataset(wsi_codes, mode="infer")
    node_features = [v.x.numpy() for idx, v in enumerate(loader)]
    node_features = np.concatenate(node_features, axis=0)
    node_scaler = StandardScaler(copy=False)
    node_scaler.fit(node_features)
    joblib.dump(node_scaler, SCALER_PATH)


def nodes_preproc_func(node_features):
    return node_scaler.transform(node_features)


# %% [markdown]
# ## The architecture holder

# %%
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear, ReLU
from torch_geometric.nn import (
    EdgeConv,
    GINConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)


class SlideGraphArch(nn.Module):
    def __init__(
        self,
        dim_features,
        dim_target,
        layers=[6, 6],
        pooling="max",
        dropout=0.0,
        conv="GINConv",
        gembed=False,
        **kwargs,
    ):
        super().__init__()
        self.dropout = dropout
        self.embeddings_dim = layers
        self.no_layers = len(self.embeddings_dim)
        self.nns = []
        self.convs = []
        self.linears = []
        self.pooling = {
            "max": global_max_pool,
            "mean": global_mean_pool,
            "add": global_add_pool,
        }[pooling]
        # if True then learn graph embedding for final classification
        # (classify pooled node features), otherwise pool node decision scores
        self.gembed = gembed

        conv_dict = {"GINConv": [GINConv, 1], "EdgeConv": [EdgeConv, 2]}
        if conv not in conv_dict:
            raise ValueError(f'Not support `conv="{conv}".')

        def create_linear(in_dims, out_dims):
            return nn.Sequential(
                Linear(in_dims, out_dims), BatchNorm1d(out_dims), ReLU()
            )

        input_emb_dim = dim_features
        out_emb_dim = self.embeddings_dim[0]
        self.first_h = create_linear(input_emb_dim, out_emb_dim)
        self.linears.append(Linear(out_emb_dim, dim_target))

        input_emb_dim = out_emb_dim
        for out_emb_dim in self.embeddings_dim[1:]:
            self.linears.append(Linear(out_emb_dim, dim_target))
            ConvClass, alpha = conv_dict[conv]
            subnet = create_linear(alpha * input_emb_dim, out_emb_dim)
            self.nns.append(subnet)
            self.convs.append(ConvClass(self.nns[-1], **kwargs))
            input_emb_dim = out_emb_dim

        self.nns = torch.nn.ModuleList(self.nns)
        self.convs = torch.nn.ModuleList(self.convs)
        # has got one more for initial input, what does this mean
        self.linears = torch.nn.ModuleList(self.linears)

    def forward(self, data):

        feature, edge_index, batch = data.x, data.edge_index, data.batch

        wsi_prediction = 0
        pooling = self.pooling
        node_prediction = 0
        for layer in range(self.no_layers):
            if layer == 0:
                feature = self.first_h(feature)
                node_prediction_sub = self.linears[layer](feature)
                node_prediction += node_prediction_sub
                node_pooled = pooling(node_prediction_sub, batch)
                wsi_prediction_sub = F.dropout(
                    node_pooled, p=self.dropout, training=self.training
                )
                wsi_prediction += wsi_prediction_sub
            else:
                feature = self.convs[layer - 1](feature, edge_index)
                if not self.gembed:
                    node_prediction_sub = self.linears[layer](feature)
                    node_prediction += node_prediction_sub
                    node_pooled = pooling(node_prediction_sub, batch)
                    wsi_prediction_sub = F.dropout(
                        node_pooled, p=self.dropout, training=self.training
                    )
                else:
                    node_pooled = pooling(feature, batch)
                    node_prediction_sub = self.linears[layer](node_pooled)
                    wsi_prediction_sub = F.dropout(
                        node_prediction_sub, p=self.dropout, training=self.training
                    )
                wsi_prediction += wsi_prediction_sub
        return wsi_prediction, node_prediction

    # running one single step
    @staticmethod
    def train_batch(model, batch_data, on_gpu, optimizer: torch.optim.Optimizer):
        wsi_graphs = batch_data["graph"].to("cuda")
        wsi_labels = batch_data["label"].to("cuda")

        # data type conversion
        wsi_graphs.x = wsi_graphs.x.type(torch.float32)

        # not RNN so does not accumulate
        optimizer.zero_grad()

        model.train()
        wsi_output, _ = model(wsi_graphs)

        # both expected to be Nx1
        wsi_labels_ = wsi_labels[:, None]
        wsi_labels_ = wsi_labels_ - wsi_labels_.T
        wsi_output_ = wsi_output - wsi_output.T
        diff = wsi_output_[wsi_labels_ > 0]
        loss = torch.mean(F.relu(1.0 - diff))

        # back prop and update
        loss.backward()
        optimizer.step()

        #
        loss = loss.detach().cpu().numpy()
        wsi_labels = wsi_labels.cpu().numpy()
        return [loss, wsi_output, wsi_labels]

    # running one single step
    @staticmethod
    def infer_batch(model, batch_data, on_gpu):
        wsi_graphs = batch_data["graph"].to("cuda")

        # data type conversion
        wsi_graphs.x = wsi_graphs.x.type(torch.float32)

        # Inference mode
        model.eval()
        # Do not compute the gradient (not training)
        with torch.inference_mode():
            wsi_output, _ = model(wsi_graphs)

        wsi_output = wsi_output.cpu().numpy()
        # Output should be a single tensor or scalar
        if "label" in batch_data:
            wsi_labels = batch_data["label"]
            wsi_labels = wsi_labels.cpu().numpy()
            return wsi_output, wsi_labels
        return [wsi_output]


# %%
wsi_codes = label_df["WSI-CODE"].to_list()
dummy_ds = SlideGraphDataset(wsi_codes, mode="infer")
loader = DataLoader(
    dummy_ds,
    num_workers=0,
    batch_size=8,
    shuffle=False,
)
iterator = iter(loader)
batch_data = iterator.__next__()

# data type conversion
wsi_graphs = batch_data["graph"]
wsi_graphs.x = wsi_graphs.x.type(torch.float32)

# %%
# ! --- [TEST] model forward integerity checking
from examples.GNN_modelling import GNN

arch_kwargs = dict(
    dim_features=2048,
    dim_target=1,
    layers=[16, 16, 8],
    dropout=0.5,
    pooling="mean",
    conv="EdgeConv",
    aggr="max",
)
pretrained = torch.load(
    "/home/dang/storage_1/workspace/tiatoolbox/local/code/data/wenqi_model.pth"
)
src_model = GNN(**arch_kwargs)
src_model.load_state_dict(pretrained)

dst_model = SlideGraphArch(**arch_kwargs)
dst_model.load_state_dict(pretrained)

src_model.eval()
dst_model.eval()
with torch.inference_mode():
    src_output, _ = src_model(wsi_graphs)
    dst_output, _ = dst_model(wsi_graphs)
    src_output = src_output.cpu().numpy()
    dst_output = dst_output.cpu().numpy()
    assert np.sum(np.abs(src_output - dst_output)) == 0
# ! ---

# %% [markdown]
# #### Training Portion

# %%
class ScalarMovingAverage(object):
    """Object to calculate running average."""

    def __init__(self, alpha=0.95):
        super().__init__()
        self.alpha = alpha
        self.tracking_dict = {}

    def __call__(self, step_output):
        for key, current_value in step_output.items():
            if key in self.tracking_dict:
                old_ema_value = self.tracking_dict[key]
                # calculate the exponential moving average
                new_ema_value = (
                    old_ema_value * self.alpha + (1.0 - self.alpha) * current_value
                )
                self.tracking_dict[key] = new_ema_value
            else:  # init for variable which appear for the first time
                new_ema_value = current_value
                self.tracking_dict[key] = new_ema_value


# %% [markdown]
# ## The running loop

# %%
from sklearn.metrics import average_precision_score as auprc_scorer
from sklearn.metrics import roc_auc_score as auroc_scorer

loader_kwargs = dict(
    num_workers=0,
    batch_size=2,
    shuffle=True,
)
arch_kwargs = dict(
    dim_features=2048,
    dim_target=1,
    layers=[16, 16, 8],
    dropout=0.5,
    pooling="mean",
    conv="EdgeConv",
    aggr="max",
)
optim_kwargs = dict(
    lr=1.0e-3,
    weight_decay=1.0e-4,
)
NUM_EPOCHS = 5


# %%
def run_once(dataset_dict, num_epochs, save_dir, on_gpu=True, pretrained=None):
    model = SlideGraphArch(**arch_kwargs)
    if pretrained is not None:
        pretrained = torch.load(pretrained)
        model.load_state_dict(pretrained)
    model = model.to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), **optim_kwargs)

    # create the graph dataset holder for each subset info then
    # pipe them through torch/torch geometric specific loader
    # for loading in multi-thread
    loader_dict = {}
    for subset_name, subset in dataset_dict.items():
        ds = SlideGraphDataset(subset, mode=subset_name, preproc=nodes_preproc_func)
        loader_dict[subset_name] = DataLoader(
            ds,
            drop_last=subset_name == "train",
            shuffle=subset_name == "train",
            **loader_kwargs,
        )

    for epoch in range(num_epochs):
        print(f"EPOCH {epoch:03d}")
        for loader_name, loader in loader_dict.items():
            # * EPOCH START
            step_output = []
            ema = ScalarMovingAverage()
            for step, batch_data in enumerate(loader):
                # * STEP COMPLETE CALLBACKS
                if loader_name == "train":
                    output = model.train_batch(model, batch_data, on_gpu, optimizer)
                    # check the output for agreement
                    ema({"loss": output[0]})
                else:
                    output = model.infer_batch(model, batch_data, on_gpu)
                    batch_size = batch_data["graph"].num_graphs
                    # iterate over output head and retrieve
                    # each as N x item, each item may be of
                    # arbitrary dimensions
                    output = [np.split(v, batch_size, axis=0) for v in output]
                    # pairing such that it will be
                    # N batch size x H head list
                    output = list(zip(*output))
                    step_output.extend(output)

            # * EPOCH COMPLETE

            # callbacks to process output
            logging_dict = {}
            if loader_name == "train":
                for val_name, val in ema.tracking_dict.items():
                    logging_dict[f"train-EMA-{val_name}"] = val
            elif loader_name == "valid":
                # expand list of N dataset size x H heads
                # back to list of H Head each with N samples
                output = list(zip(*step_output))
                prob, true = output
                prob = np.squeeze(np.array(prob))
                true = np.squeeze(np.array(true))

                val = auroc_scorer(true, prob)
                logging_dict["valid-auroc"] = val
                val = auprc_scorer(true, prob)
                logging_dict["valid-auprc"] = val

            # callbacks for logging and saving
            for val_name, val in logging_dict.items():
                print(f"{val_name}: {val}")
            if "train" not in loader_dict:
                continue

            # track the statistics
            new_stats = {}
            if os.path.exists(f"{save_dir}/stats.json"):
                old_stats = load_json(f"{save_dir}/stats.json")
                # save a backup first
                save_as_json(logging_dict, f"{save_dir}/stats.old.json")
                new_stats = copy.deepcopy(old_stats)

            old_epoch_stats = {}
            if epoch in old_epoch_stats:
                old_epoch_stats = new_stats[epoch]
            old_epoch_stats.update(logging_dict)
            new_stats[epoch] = old_epoch_stats
            save_as_json(new_stats, f"{save_dir}/stats.json")

            # save the pytorch model
            torch.save(model.state_dict(), f"{save_dir}/epoch={epoch:03d}-weights.pth")
    return step_output


# %% [markdown]
# ### The training portion


# %%
ROOT_OUTPUT_DIR = "/home/dang/storage_1/workspace/tiatoolbox/local/code/"
MODEL_DIR = f"{ROOT_OUTPUT_DIR}/model/"
for split_idx, split in enumerate(split_list):
    split_ = copy.deepcopy(split)
    split_.pop("test")
    split_save_dir = f"{MODEL_DIR}/{split_idx:02d}/"
    rm_n_mkdir(split_save_dir)
    run_once(split_, NUM_EPOCHS, split_save_dir)


# %% [markdown]
# ### The inference portion

# %% [markdown]
# #### The model selections

# %%
PRETRAINED_DIR = "/home/dang/storage_1/workspace/tiatoolbox/local/code/model/"
stat_files = recur_find_ext(PRETRAINED_DIR, [".json"])
stat_files = [v for v in stat_files if ".old.json" not in v]


# %%


def select_checkpoints(stat_file_path, top_k=2, metric="valid-auprc"):
    stats_dict = load_json(stat_file_path)
    # k is the epoch counter in this case
    stats = [[int(k), v[metric]] for k, v in stats_dict.items()]
    # sort epoch ranking from largest to smallest
    stats = sorted(stats, key=lambda v: v[1], reverse=True)
    chkpt_stats_list = stats[:top_k]  # select top_k

    model_dir = pathlib.Path(stat_file_path).parent
    epochs = [v[0] for v in chkpt_stats_list]
    paths = [f"{model_dir}/epoch={epoch:03d}-weights.pth" for epoch in epochs]
    return paths, chkpt_stats_list


chkpt_paths, chkpt_stats_list = select_checkpoints(stat_files[0])

# %% [markdown]
# #### Bulk inference and ensemble results

cum_results = []
# for split_idx, split in enumerate(split_list):

split = copy.deepcopy(split_list[0])
for chkpt_path in chkpt_paths:
    split_ = {"infer": [v[0] for v in split["test"]]}
    chkpt_results = run_once(
        split_,
        num_epochs=1,
        save_dir=None,
        pretrained=chkpt_path,
    )
    cum_results.append(chkpt_results)
cum_results = np.array(cum_results)
cum_results = np.squeeze(cum_results)
