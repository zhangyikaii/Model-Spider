import glob
import os
import os.path as osp

from datasets.udomain import Datum, DatasetBase


def listdir_nohidden(path, sort=False):
    """List non-hidden items in a directory.

    Args:
         path (str): directory path.
         sort (bool): sort the items.
    """
    items = [f for f in os.listdir(path) if not f.startswith(".")]
    if sort:
        items.sort()
    return items


class VLCS(DatasetBase):
    """VLCS.

    Statistics:
        - 4 domains: CALTECH, LABELME, PASCAL, SUN
        - 5 categories: bird, car, chair, dog, and person.

    Reference:
        - Torralba and Efros. Unbiased look at dataset bias. CVPR 2011.
    """

    dataset_dir = "VLCS"
    domains = ["caltech", "labelme", "pascal", "sun"]
    data_url = "https://drive.google.com/uc?id=1r0WL5DDqKfSPp9E3tRENwHaXNs1olLZd"

    def __init__(self, root, source_domain, target_domain):
        root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.check_input_domains(
            source_domain, target_domain
        )

        train = self._read_data(source_domain, "train")
        val = self._read_data(source_domain, "crossval")
        test = self._read_data(target_domain, "test")

        super().__init__(train_x=train, val=val, test=test)

    def _read_data(self, input_domains, split):
        items = []

        for domain, dname in enumerate(input_domains):
            dname = dname.upper()
            path = osp.join(self.dataset_dir, dname, split)
            folders = listdir_nohidden(path)
            folders.sort()

            for label, folder in enumerate(folders):
                impaths = glob.glob(osp.join(path, folder, "*.jpg"))

                for impath in impaths:
                    item = Datum(impath=impath, label=label, domain=domain)
                    items.append(item)

        return items
