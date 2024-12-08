# Camel Morph

[![License](https://img.shields.io/github/license/CAMeL-Lab/camel_morph)](https://opensource.org/licenses/MIT)

The work presented in this repository is part of a large effort on Arabic morphology under the name of the Camel Morph Project [^1] developed by the [CAMeL Lab](http://camel-lab.com/) at [New York University Abu Dhabi](http://nyuad.nyu.edu/).

> **Please use** [GitHub Issues](https://github.com/CAMeL-Lab/camel_morph/issues) **to report a bug or if you need help using Camel Morph.**

Camel Morph’s goal is to build large open-source morphological models for Arabic and its dialects across many genres and domains. This repository contains code meant to build an ALMOR-style database (DB) from a set of morphological specification and lexicon spreadsheets, which can then be used by [Camel Tools](https://github.com/CAMeL-Lab/camel_tools) for morphological analysis, generation, and reinflection.

<p align="center"> <img width="70%" height="70%" src="camel-morph-system.jpg"> </p>

The following sections provide useful usage information about the repository. 

<!-- For pointers to the system used for the SIGMORPHON 2022 Camel Morph paper, check the [Camel Morph SIGMORPHON 2022](#camel-morph-sigmorphon-2022) section. -->

## Official Camel Morph DB Releases

* camel_morph_msa_v1.0.db [file](./official_releases/lrec-coling2024_release/databases/camel-morph-msa/camel_morph_msa_v1.0.db) (LREC-COLING 2024 release).
  To cite this release use Khairallah et al. (2024).[^1]
  

## Repositories of Published Efforts
The work has been reported on in three papers (see below), but is continuously updtaed.

### Camel Morph MSA LREC-COLING 2024

For instructions related to inspecting, making use of, replicating the results obtained for the LREC-COLING  2024 Camel Morph MSA full database paper,[^1] and the data, see the [official_releases/lrec-coling2024_release/](./official_releases/lrec-coling2024_release/) folder.

### Camel Morph Nominals EACL 2024

For instructions related to inspecting, making use of, replicating the results obtained for the EACL 2024 Camel Morph Nominals paper,[^2] and the data, see the [official_releases/eacl2024_release/](./official_releases/eacl2024_release/) folder.

### Camel Morph Verbs SIGMORPHON 2022

For instructions related to inspecting, making use of, replicating the results obtained for the SIGMORPHON 2022 Camel Morph paper,[^3] and the data, see the [official_releases/sigmorphon2022_release/](./official_releases/sigmorphon2022_release/) folder.

## Data

[![License](https://mirrors.creativecommons.org/presskit/buttons/80x15/svg/by.svg)](https://creativecommons.org/licenses/by/4.0/)

The data throughout this project is being maintained through the Google Sheets interface which can be used to add, delete, or edit morphological specification entries. The following are links to the data and morphological specifications used for this project, and are **only accessible upon demand**.

### Data License

The data files accessed through the below links are licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/). For code license, see [License](#license).

### Continuously Updated Data
- Latest MSA Camel Morph db [file](./official_releases/lrec-coling2024_release/databases/camel-morph-msa/camel_morph_msa_v1.0.db) (LREC-COLING 2024 release) 
- [MSA Verbs Specifications](https://docs.google.com/spreadsheets/d/1thVU-IP-I-XnOmy5XfdUr39eNgDldZfMeTI75EH1TjQ/edit?usp=sharing)
- [EGY Verbs Specifications](https://docs.google.com/spreadsheets/d/1NBK_UPl2799GRBkOrd9Eub_yLn2U2ccbJqYETKdCcbw/edit?usp=sharing)
- [MSA Nominals and Others Specifications](https://docs.google.com/spreadsheets/d/1QsIZ8ToFovoltyfIrUtStxrB_G8jWbj01aHgQQ9IIwE/edit#gid=337793670)


### LREC-COLING 2024 Data (frozen)

The data is accessible from the following [folder](./official_releases/lrec-coling2024_release/data/).

<!-- ### LREC-COLING 2024 Data (frozen)

The following data is not accessile publicly from the Google Sheets interface but is available in `csv` format (the way it was at submission time) in the following [folder](./official_releases/lrec-coling2024_release/data/).

- [MSA Verbs Specifications (Camera Ready)](https://docs.google.com/spreadsheets/d/1V6TdCM6V5byu9HGCdmVY979MhQ2pyNQdO8XkRx3_n2M/edit#gid=210443809)
- [MSA Nominals and Others Specifications (Camera Ready)](https://docs.google.com/spreadsheets/d/1s3nocf4bAxOsXjcvSMulJr5N9Yq1yUWyy5M6XkJk2_s/edit#gid=898723826)
- [MSA Annex - Wiki Proper Nouns (Camera Ready)](https://docs.google.com/spreadsheets/d/1U_V8wNo5gHokTdxG5HaEaqcgjArgecRLiXEi5kMIlX4/edit#gid=1328530526) -->

### EACL 2024 Data (frozen)

The following data is not accessile publicly from the Google Sheets interface but is available in `csv` format (the way it was at submission time) in the following [folder](./official_releases/eacl2024_release/data/).

- [MSA Nominal Specifications (Camera Ready)](https://docs.google.com/spreadsheets/d/1T5-tY_bfvCW579P-NY7nOxxlq4P2MQ7vmWg_IWn_35Q/edit#gid=1109514510)

### SIGMORPHON 2022 Data (frozen)

The following data is not accessile publicly from the Google Sheets interface but is available in `csv` format (the way it was at submission time) in the following [folder](/official_releases/sigmorphon2022_release/data/).

- [MSA Verbs Specifications (Camera Ready)](https://docs.google.com/spreadsheets/d/1v9idxctnr6IsqG4c7bHs7lGx7GzbnTa2s4ghQCmLoPY/edit#gid=524706154)
- [EGY Verbs Specifications (Camera Ready)](https://docs.google.com/spreadsheets/d/1OCqHIdeZpm9BNa-BiC7Xy6bAT_wkLnhuvKdo7X3-RtE/edit#gid=424095452)


## Folder Hierarchy

The following table describes the function of each directory contained in the repository.

| Directory | Description |
| ----------- | ----------- |
| `./camel_morph` | Directory (in package format) containing all the necessary files to build, debug, test, and evaluate the Camel Morph DB Maker. |
| `./camel_morph/configs` | Contains configuration files which make running the scripts in the above directory easier.
| `./data` | Contains, for each different configuration, the set of morphological specification files necessary to run the different scripts. This directory is mandatorily (as per the data reader code) organized into project directories as described in [Configuration File Structure](#configuration-file-structure) section.
| `./databases` | Contains the output DB files resulting from the DB Making process.
| `./misc_files` | Contains miscellaneous files used by scripts inside `./camel_morph`.
| `./official_releases/sigmorphon2022_release` | Standalone environment[^4] allowing users to run the DB Maker and Camel Tools engines without installing Camel Tools, in the same version used for the SIGMORPHON 2022 paper. Also contains the data that was used to get the results described in the paper[^3].

## Instructions

To compile databases, paradigm-specific inflection (conjugation/declension) tables, or evaluation tables, follow the below instructions. A default configuration file is included in the `./camel_morph/configs` directory for direct usage.

### Installation

To start working with the Camel Morph environment and compiling Mordern Standard Arabic (MSA) databases:

1. Clone (download) this repository and unzip in a directory of your choice.
2. Make sure that you are running **Python 3.8** or **Python 3.9** (this release was tested on these two versions, but it is likely that it will work on other versions).
3. Run the following command to install all needed libraries: `pip install -r requirements.txt`.
4. Run all commands/scripts from the outer `camel_morph` directory.

#### For development purposes only

To debug and evaluate databases (MSA or Dialectal Arabic), and for other utilities:

1. Clone (download) a [fork](https://github.com/christios/camel_tools) of the Camel Tools repository. The Camel Morph databases will currently only function using the latter instance of Camel Tools. The changes in this fork will eventually be integrated to the main Camel Tools library. Unzip in a directory of your choice.
2. Set the `$CAMEL_TOOLS_PATH` value to the path of the Camel Tools fork repository in the configuration file that you will be using (default configuration file `./camel_morph/configs/config_default.json` provided; see [Configuration File Structure](#configuration-file-structure) section).

For instructions on how to run the different scripts, see the below sections.

### Compiling a Database (DB Maker)

The below command compiles an ALMOR-style database starting from a set of morphological specification files referenced in the specific configuration mentioned as an argument. Before starting compilation, the specifications should be downloaded from the links provided in the data section.

#### Usage

```bash
usage: db_maker.py [-h] [-config_file CONFIG_FILE]
                   [-config_name CONFIG_NAME]
                   [-output_dir OUTPUT_DIR]
                   [-run_profiling]
                   [-camel_tools {local,official}]
```

#### Arguments

|short|default|help|
| :--- | :--- | :--- |
| `-h` | | Show this help message and exit.|
|`-config_file`|`config_default.json`|Name of the configuration file which contains different configurations to run the DB on, and which should be contained in the `./camel_morph/configs/` directory. Some pre-compiled configurations already exist in `./camel_morph/configs/config_default.json`, but new ones could be easily added. See [here](#configuration-file-structure) for an overview of the configuration file format. Defaults to `config_default.json`.|
|`-config_name`|`default_config`|Configuration name of one of the configurations contained in `CONFIG_FILE`. It contains script parameters, sheet paths, etc.|
|`-output_dir`||Overrides path of the directory to output the DBs to (specified in the global section of `CONFIG_FILE`).|
|`-run_profiling`||To generate an execution time profile of the specific configuration.|
|`-camel_tools`|`local`|Path of directory containing the CAMeL Tools modules (should be cloned as described [here](#for-development-purposes-only)).|

### Utilities

There are various scripts in the suite which are meant to make the debugging/evaluation experience more efficient. To be able to make use of those, many require a (free) service account to be created using Google Cloud, to get an API key (service account) to add to our internal configuration files for use. Google Cloud will generate a JSON file which should be stored locally, and the path of which should be specified in the `global` section of the [configuration](#default-configuration-file) as follows: `"service_account": $SERVICE_ACCOUNT_PATH`.

Follow the instructions until minute 2:00 of [this](https://www.youtube.com/watch?v=fxGeppjO0Mg) video to first create a service account and API key to use with the Google Sheets API, and then [this](https://www.youtube.com/watch?v=rWcLDax-VmM) video to generate the JSON object referred to in the previous paragraph.

### Configuration File Structure

In its most basic format, the configuration file should look like the example below in order to successfully run the scripts described in this guide. Unless otherwise stated, variables (beginning with `$`) are double quoted strings. See [here](camel_morph/configs/) for a list of configuration files used. Also, note that the configuration file can include many other keys/values that are useful for debugging purposes, as specified by the `Config` reader [class](https://github.com/CAMeL-Lab/camel_morph/blob/fe3242037ea45b348e9950c6e6cf9aa46cf9209d/camel_morph/utils/utils.py#L527).

#### Default Configuration File

    {
        "global": {
            "data_dir": $DATA_DIR_PATH,
            "specs": {
                "about": {
                    $SPREADSHEET_X: $ABOUT_SHEET,
                },
                "header": {
                    $SPREADSHEET_X: $HEADER_SHEET,
                }
            },
            "db_dir": $DB_OUTPUT_DIR,
            "camel_tools": $CAMEL_TOOLS_PATH
        },
        "local": {
            $CONFIG_NAME: {
                "dialect": $DIALECT,
                "cat2id": $CAT2ID,
                "reindex": $REINDEX,
                "pruning": $PRUNING,
                "specs": {
                    "order": {
                        $SPREADSHEET_X: $ORDER_SHEET_1,
                        $SPREADSHEET_X: $ORDER_SHEET_2,
                        ...
                    },
                    "morph": {
                        $SPREADSHEET_X: $MORPH_SHEET_1,
                        $SPREADSHEET_X: $MORPH_SHEET_2,
                        ...
                    },
                    "lexicon": {
                        $SPREADSHEET_X: $LEXICON_SHEET_1,
                        $SPREADSHEET_X: $LEXICON_SHEET_2,
                        ...
                    }
                },
                "db": $DB_NAME,
                "pos_type": $POS_TYPE,
                "class_map": $CLASS_MAP
            }
        }
    }

where:

- `$DATA_DIR_PATH`: path of the outermost data directory where all sheets are kept (e.g., `data`; referenced from the main repository directory). Sheets for this configuration should be kept inside a folder which has the name as the configuration (`$CONFIG_NAME`) which is itself contained in a directory called `camel-morph-$DIALECT` (where `$DIALECT` is specified below). So for example, if `$DATA_DIR_PATH=data`, `$DIALECT=msa`, and `$CONFIG_NAME=pv_msa`, then sheets for this configuration should be in a directory with the path `./data/camel-morph-msa/pv_msa`.
- `$SPREADSHEET_X`: name of the the spreadsheet on Google Sheets containing the sheet which is assigned to it as a value. If no spreadsheet is associated with the sheet, just keep blank.
- `$ABOUT_SHEET`: name of the sheet containing the *About* section which will go in the DB (e.g., `About`). Either downloaded automatically as specified in the [Utilities](#utilities) section or manually.
- `$HEADER_SHEET`: same as `$ABOUT_SHEET` (e.g., `Header`)
- `$DB_OUTPUT_DIR`: name of the directory to which the compiled DBs will be output.
- `$CAMEL_TOOLS_PATH`: path of the Camel Tools repository fork that should be cloned/downloaded as described in [Installation](#installation) section.
- `$CONFIG_NAME`: name of the configuration in the `local` section of the config file, to choose between a number of different configurations (e.g., `default_config`). This is also the name of the folder which contains the sheets that are specified for that configuration and the global section.
- `$DIALECT`: dialect being worked with (i.e., `msa` or `egy`). This is specified to further organize the configuration-specific data into high-level projects (i.e., `./data/camel-morph-msa` or `./data/camel-morph-egy`).
- `$CAT2ID`: boolean (`true` or `false`). Specifies the format in which to output the ALMOR morpheme category names. If set to true, then category names are IDs, otherwise, they contain condition information.
- `$PRUNING`: boolean (`true` or `false`). Used in the DB making process to speed up DB compilation. For this to be set to `true`, the Morph sheet must contain condition definitions (organization of conditions into categories).
- `$REINDEX`: boolean (`true` or `false`). Used in the DB making process to collapse categories after the entries are compiled. This heavily reduces the size of the compatibility tables, and turns category names into compact unique IDs (basically, doing what `cat2id` does and more).
- `$ORDER_SHEET`: same as `$ABOUT_SHEET` (e.g., `MSA-Verb-ORDER`).
- `$MORPH_SHEET`: same as `$ABOUT_SHEET` (e.g., `MSA-Verb-MORPH`).
- `$LEX_SHEET_1`: same as `$ABOUT_SHEET` (e.g., `MSA-Verb-LEX-PV`). At least one lexicon sheet can be specified; the latter will be concatenated in pre-processing.
- `$DB_NAME`: name of the output DB.
- `$POS_TYPE`: type of the POS for which we are building the DB. Can either be `verbal`, `nominal` or `any`. As far as the DB Maker is concerned this controls what MADA features are output to the DB file in each line.
- `$CLASS_MAP`: dictionary containing the morpheme classes and which complex morpheme type they map to.

## Code License

All the code contained in this repository is available under the MIT license. See the [LICENSE](./LICENSE) file for more info.

## Contributors

- [Christian Khairallah (Cayralat)](https://github.com/christios)
- [Muhammed AbuOdeh](https://github.com/muhammed-abuodeh)
- [Salam Khalifa](https://github.com/slkh)
- [Reham Marzouk](https://scholar.google.com.eg/citations?user=VGVUg3cAAAAJ&hl=en)
- [Nizar Habash](https://github.com/nizarhabash1)



[^1]: Khairallah, Christian, Salam Khalifa, Reham Marzouk, Mayar Mohamadein Nassar and Nizar Habash. [Camel Morph MSA: A Large-Scale Open-Source Morphological Analyzer for Modern Standard Arabic](https://aclanthology.org/2024.lrec-main.240.pdf). In Proceedings of the LREC-COLING 2024 - The Joint International Conference on Computational Linguistics, Language Resources and Evaluation, Turin, Italy. 2024.

[^2]: Khairallah, Christian, Reham Marzouk, Salam Khalifa, Mayar Nassar, and Nizar Habash. [Computational Morphology and Lexicography Modeling of Modern Standard Arabic Nominals](https://aclanthology.org/2024.findings-eacl.72.pdf). In Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics, Malta, 2024.

[^3]: Nizar Habash, Reham Marzouk, Christian Khairallah, and Salam Khalifa. 2022. [Morphotactic Modeling in an Open-source Multi-dialectal Arabic Morphological Analyzer and Generator](https://aclanthology.org/2022.sigmorphon-1.10/). In Proceedings of the 19th SIGMORPHON Workshop on Computational Research in Phonetics, Phonology, and Morphology, pages 92–102, Seattle, Washington. Association for Computational Linguistics.

[^4]: Note that for the release directory, only the morphological components from Camel Tools were sourced from the actual library and were added to be imported locally.
