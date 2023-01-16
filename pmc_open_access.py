"""Huggingface dataset loader for PubMed Central Open Access dataset.

based on https://huggingface.co/datasets/albertvillanova/pmc_open_access
"""


import csv
import io
from pathlib import Path
import tarfile

import datasets
import pandas as pd


# PMC Open Access Subset [Internet]. Bethesda (MD): National Library of Medicine. 2003 - [cited YEAR MONTH DAY]. Available from https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/.
_CITATION = """\
@misc{pmc-open-access-subset,
  url = {https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/},
  title = {PMC Open Access Subset},
  publisher = {National Library of Medicine},
  year = {2023}
}
"""

_DESCRIPTION = """\
The PMC Open Access Subset includes millions of journal articles and preprints that are made available under license terms that allow reuse. Not all articles in PMC are available for text mining or other reuse; many are under copyright. Articles in the PMC Open Access Subset are made available under Creative Commons or similar licenses that allow more liberal redistribution and reuse than a traditionally copyrighted work. The PMC Open Access Subset is one part of the PMC Article Datasets.
"""

_HOMEPAGE = "https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/"

_LICENSE = ""

_SUBSETS = {
    "commercial": "oa_comm",
    "non_commercial": "oa_noncomm",
    "other": "oa_other",
}
_BASELINE_DATE = "2022-12-17"


class PmcOpenAccess(datasets.GeneratorBasedBuilder):
    """PMC Open Access Subset."""

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "pmid": datasets.Value("string"),
                    "accession_id": datasets.Value("string"),
                    "license": datasets.Value("string"),
                    "last_updated": datasets.Value("string"),
                    "retracted": datasets.Value("string"),
                    "citation": datasets.Value("string"),
                    "decoded_as": datasets.Value("string"),
                    "journal": datasets.Value("string"),
                    "year": datasets.Value("int32"),
                    "doi": datasets.Value("string"),
                    "oa_subset": datasets.Value("string"),
                }
            ),
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):

        if self.config.data_dir is None:
            raise ValueError(
                "This loader requires a local path. "
                "Please pass the path to oa_bulk using the data_dir kwarg."
            )
        else:
            data_dir = self.config.data_dir

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_dir": data_dir,
                    "split": str(datasets.Split.TRAIN),
                },
            ),
        ]

    def gen_from_tar_and_filelist_paths(self, tar_paths, filelist_paths, dict_pmc_ids):

        for tar_path, filelist_path in zip(tar_paths, filelist_paths):
            df_filelist = pd.read_csv(filelist_path)
            dicts_filelist = df_filelist.to_dict(orient="records")

            with tarfile.open(tar_path, "r") as tf:
                for ii, tarinfo in enumerate(tf):
                    meta = dicts_filelist[ii]
                    fp = tf.extractfile(tarinfo)
                    content_bytes = fp.read()

                    try:
                        content_text = content_bytes.decode("utf-8")
                        meta["decoded_as"] = "utf-8"
                    except UnicodeDecodeError as e:
                        context_text = content_bytes.decode("latin-1")
                        meta["decoded_as"] = "latin-1"

                    pmc_id = meta["AccessionID"]
                    more_meta = dict_pmc_ids.get(pmc_id)
                    if more_meta is None:
                        journal = None
                        year = None
                        doi = None
                    else:
                        journal = more_meta["Journal Title"]
                        year = more_meta["Year"]
                        doi = more_meta["DOI"]

                    sample = {
                        "text": content_text,
                        "pmid": meta["PMID"],
                        "accession_id": pmc_id,
                        "license": meta["License"],
                        "last_updated": meta["LastUpdated (YYYY-MM-DD HH:MM:SS)"],
                        "retracted": meta["Retracted"],
                        "citation": meta["Article Citation"],
                        "decoded_as": meta["decoded_as"],
                        "journal": journal,
                        "year": year,
                        "doi": doi,
                    }
                    yield sample

    def _generate_examples(self, data_dir, split):

        data_dir = Path(data_dir)

        dict_pmc_ids = {
            row["PMCID"]: row
            for row in pd.read_csv(data_dir / "PMC-ids.csv.gz").to_dict(
                orient="records"
            )
        }

        _id = 0

        for subset in _SUBSETS.values():

            base_path = data_dir / "oa_bulk" / subset / "txt"

            incremental_tar_paths = sorted(list(base_path.glob("*incr*.tar.gz")))
            incremental_filelist_paths = sorted(
                list(base_path.glob("*incr*.filelist.csv"))
            )
            assert len(incremental_tar_paths) == len(incremental_filelist_paths)
            for sample in self.gen_from_tar_and_filelist_paths(
                incremental_tar_paths, incremental_filelist_paths, dict_pmc_ids
            ):
                sample["oa_subset"] = subset
                yield _id, sample
                _id += 1


            baseline_tar_paths = sorted(list(base_path.glob("*baseline*.tar.gz")))
            baseline_filelist_paths = sorted(
                list(base_path.glob("*baseline*.filelist.csv"))
            )
            assert len(baseline_tar_paths) == len(baseline_filelist_paths)
            for sample in self.gen_from_tar_and_filelist_paths(
                baseline_tar_paths, baseline_filelist_paths, dict_pmc_ids
            ):
                sample["oa_subset"] = subset
                yield _id, sample
                _id += 1
