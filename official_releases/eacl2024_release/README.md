# Camel Morph EACL 2024

This section guides you through the process of inspecting, making use of, and replicating the results obtained for the EACL 2024 Camel Morph paper. All the data and code (including relevant Camel Tools modules) required to replicate the paper results are already contained in the standalone `camel_morph/official_releases/eacl2024_release/` directory. Furthermore, the generated DB can be read using the official [Camel Tools](https://github.com/CAMeL-Lab/camel_tools) release, but the evaluation/statistics scripts can only ran using the Camel Tools modules included in the latter directory. To replicate the paper results, follow the below instructions.

## Installation

To start working with the Camel Morph environment and compiling Mordern Standard Arabic (MSA) databases:

1. Clone (download) this repository and unzip in a directory of your choice.
2. Make sure that you are running **Python 3.8** or **Python 3.9** (this release was tested on these two versions, but it is likely that it will work on other versions).
3. Run the following command to install all needed libraries: `pip install -r requirements.txt`.
4. Run all commands/scripts from the `eacl2024_release/` directory.

## Data

[![License](https://mirrors.creativecommons.org/presskit/buttons/80x15/svg/by.svg)](https://creativecommons.org/licenses/by/4.0/)

The data used to compile a database is available in `csv` format (the way it was at submission time) in the [data/](./data/) folder contained in this directory.

The data is also available for viewing from the Google Sheets interface using the following links, although, column width is not adjustable (you can download the sheet and open it in a personal spreadsheet as a workaround).

- [MSA Nominals Specifications (Camera Ready)](https://docs.google.com/spreadsheets/d/1T5-tY_bfvCW579P-NY7nOxxlq4P2MQ7vmWg_IWn_35Q/edit#gid=1109514510)

### Data License

The data files accessed through the below links are licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/). For code license, see [License](#license).

## Modern Standard Arabic (MSA) Nominals Results

### Generating a DB

To generate the MSA nominals database, the results of which were described in the paper, run the following two commands from the main repository directory to output the resulting [DB](./databases/camel-morph-msa/msa_nom_eacl2024.db) into the `eacl2024_release/databases/camel-morph-msa` directory. From the main `camel_morph/` folder, run:

    >> cd eacl2024_release
    >> python camel_morph/db_maker.py -config_file config_nominals.json -config_name nominals_msa 

### Generating the Paper Statistics

To generate the statistics found in *Table 5* of the paper, run:

    >> cd eacl2024_release
    >> python camel_morph/eval/evaluate_camel_morph_stats.py -config_file config_nominals.json -config_name nominals_msa -msa_baseline_db <calima_msa_db_path> -no_download -no_build

where you would need the `calima-msa-s31_0.4.2.utf8.db` DB file which is stored [here](https://drive.google.com/file/d/1ggbUpaXJ_-jiGhmpGsMRpd9SwM0wZo17/view?usp=drive_link), but which is not publicly accessible because it is under copyrights.

Finally, to get the results in Appendix C (Nominal Lemmas Paradigm Index), run:

    >> cd eacl2024_release
    >> python camel_morph/eval/evaluate_camel_morph_stats.py -config_file config_nominals.json -config_name nominals_msa -lemma_paradigm_sheet data/camel-morph-msa/nominals_msa/Lemma-Paradigm-Reference-Nom.csv -lemma_paradigm_sheet_paper_asset data/camel-morph-msa/nominals_msa/Lemma-Paradigms-Index-v1.1.csv -no_download -no_build

### Evaluation

To evaluate and get the results presented in the *Evaluation* section of the paper, then run:

    >> cd eacl2024_release
    >> python camel_morph/eval/evaluate_camel_morph.py -eval_mode recall_msa_magold_ldc_dediac_match_no_null -config_file config_nominals.json -msa_config_name nominals_msa_eval -pos_or_type <required_pos> -output_dir <report_dir> -msa_magold_path <msa_magold_path>

where `<required_pos>` should be either `noun`, `adj`, or `adj_comp`; `<report_dir>` is the folder path where you want to output the evaluation report to, e.g., `eacl2024_eval/`. Also, to run this script you would need the Calima-synchronized ATB data for the `-msa_magold_path` argument, which is [here](https://drive.google.com/file/d/1mVWONav2pxIdwBTJQaZovGpUqUIe3eBa/view?usp=drive_link), but also not publicly accessible.

## Analysis and Generation

In order to use the generated DB for analysis or generation, follow the same instructions provided in the examples at the following links:

- [Analysis](https://camel-tools.readthedocs.io/en/latest/api/morphology/analyzer.html)
  - [Disambiguation](https://camel-tools.readthedocs.io/en/latest/api/disambig/mle.html) (in-context analysis)
- [Generation](https://camel-tools.readthedocs.io/en/latest/api/morphology/generator.html)
