from schnetpack.datasets import QM9
import schnetpack as spk
import os

qm9data = QM9('./data/qm9.db', download=True, load_only=[QM9.U0])
qm9split = './data/qm9split'

train, val, test = spk.train_test_split(
    data=qm9data,
    num_train=30000,
    num_val=10000,
    split_file=os.path.join(qm9split, "split30000-10000.npz"),
)