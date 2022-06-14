# CamelMorph

## Introduction

The work presented in this repository is part of a large effort on Arabic morphology under the name of the CamelMorph Project [^2]. CamelMorph’s goal is to build large open-source morphological models for Arabic and its dialects across many genres and domains. This repository contains code meant to build an ALMOR-style database from a set of morphological specification and lexicon spreadsheets, which can then be used by [Camel Tools](https://github.com/CAMeL-Lab/camel_tools)’s for morphological analysis, generation, and reinflection.

The following sections provide useful usage information about the repository. For pointers to the system used for the SIGMORPHON 2022 CamelMorph paper, check the [CamelMorph SIGMORPHON 2022](#camelmorph-sigmorphon-2022) section.

## CamelMorph SIGMORPHON 2022

This section guides you through the process of inspecting, making use of, and replicating the results obtained for the SIGMORPHON 2022 CamelMorph paper[^2]. Firstly, all the data can be 

## Google Sheets

The data throughout this project is being managed through the Google Sheets interface, through which we can add, delete, or edit entries. TODO (add links)

## Installation

To start working with the CamelMorph engine:

1. Clone this repository and a [fork](https://github.com/christios/camel_tools) of the Camel Tools repository.
2. Unzip both in a directory of your choice.
3. Make sure that the following are installed using `pip install`: **Python 3.3** or newer, **NumPy 1.20** or newer, and **Pandas 1.4** or newer.
4. Run all commands/scripts from the outer `camel_morph` directory.

For instructions on how to run the different scripts, see the [Instructions](#instructions) section.

## Folder Hierarchy

| Directory | Description |
| ----------- | ----------- |
| `./camel_morph` | Python package containing all the necessary files to build, debug, test, and evaluate. |
| `./configs` | Contains configuration files which make running the scripts in the above directory easier.
| `./data` | Contains, for each different configuration, the set of morphological specification files necessary to run the different scripts.
| `./databases` | Contains the output DB files resulting from the DB Making process.
|`./debugging_output` | Contains all DB debugging files which are usually uploaded (automatically) to and analyzed in Google Sheets.
| `./eval_files` | Contains files necessary to carry out evalutation of generated DBs and analyzable evaluation output files (generally analyzed on Google Sheets).
| `./misc_files` | Contains miscellaneous files used by scripts inside `./camel_morph`.
| `./sandbox` | Contains various standalone files that are not used in `./camel_morph`.

## Instructions

To generate databases, paradigm-specific inflection (conjugation/declension) tables, or evaluation tables, follow the below instructions. Note that the adjoining `Makefile` can be used to reproduce the experiments which were ran in-house by providing the (pipelines of) instructions that were used to carry them out. But first, it is important to understand the structure of the configuration file. Also, a default configuration file is included in the `./configs` directory for direct usage.

### Configuration File Structure

In its most basic format, the configuration file should look like the following:

    {
        "global": {
            "data_dir": DATA_DIR_PATH,
            "specs": {
                "sheets": [
                    ABOUT-SHEET,
                    HEADER-SHEET
                ]
            },
            "db_dir": DB_OUTPUT_DIR,
            "camel_tools": CAMEL_TOOLS_PATH
        },
        "local": {
            CONFIG_NAME: {
                "dialect": DIALECT,
                "pruning": PRUNING,
                "specs": {
                    "sheets": {
                        "order": ORDER-SHEET,
                        "morph": MORPH-SHEET
                    }
                },
                "lexicon": {
                    "sheets": [
                        LEX-SHEET-1,
                        ...
                    ]
                },
                "db": DB_NAME,
                "pos_type": POS_TYPE
            }
        }
    }

where (content of placeholders below should be enclosed in double quotes):

- `DATA_DIR_PATH`: path of the outermost data directory where all sheets are kept referenced from the outermost `camel_morph` directory (e.g., `data`)
- `ABOUT-SHEET`: name of the sheet containing the *About* section which will go in the DB (e.g., `About`). Downloaded as specified in the [Google Sheets](#google-sheets) section.
- `HEADER-SHEET`: same as `ABOUT-SHEET` (e.g., `Header`)
- `DB_OUTPUT_DIR`: name of the directory to which the compiled DBs will be output.
- `CAMEL_TOOLS_PATH`: path of the Camel Tools repository fork that should be cloned/downloaded as described in [Installation](#installation) section.
- `CONFIG_NAME`: name of the configuration in the `local` section of the config file, to choose between a number of different configurations (e.g., `default_config`). This is also the name of the folder which contains the sheets that are specified for that configuration and the global section.
- `DIALECT`: dialect being worked with (i.e., `msa` or `egy`). This is specified to further organize the configuration-specific data into high-level projects (i.e., `./data/camel-morph-msa` or `./data/camel-morph-egy`).
- `PRUNING`: boolean (`true` or `false`). Used in the DB making process to speed up DB compilation. For this to be set to `true`, the Morph sheet must contain condition definitions (organization of conditions into categories).
- `ORDER-SHEET`: same as `ABOUT-SHEET` (e.g., `MSA-Verb-ORDER`).
- `MORPH-SHEET`: same as `ABOUT-SHEET` (e.g., `MSA-Verb-MORPH`).
- `LEX-SHEET-1`: same as `ABOUT-SHEET` (e.g., `MSA-Verb-LEX-PV`). At least one lexicon sheet can be specified; the latter will be concatenated in pre-processing.
- `DB_NAME`: name of the output DB.
- `POS_TYPE`: type of the POS for which we are building the DB. Can either be `verbal` or `nominal`.

### Downloading Files

To download files from Google Drive, first, follow the instructions in the first 3:50 minutes of [this](https://www.youtube.com/watch?v=bu5wXjz2KvU) video to get API access to the Google Drive interface. Then, run the following command to download selected/individual sheets (not spreadsheets) in csv format from the following cloud [directory](https://drive.google.com/drive/folders/1yRq5PZ7rwQKzGCIIcoVPvgbTLHrkkxpE).

    python download_sheets.py [-save_dir A] [-config_file B -config_name C] [-lex D1 D2 [...] [-lex ...] -specs E1 E2 [...] [-specs ...]]

- `A`: path of the directory to save all the downloaded files in. It is `./data` by default (if it does not exist, it will be automatically created).
- `B`: path of the configuration file (already included in repo, i.e., `./config.json`) which contains different configurations to run the DB on (previsouly the `PARAMS` sheet in the specs spreadsheet). Some pre-compiled configurations already exist in `config.json`, but new ones could be easily added on demand.
- `C`: configuration name of one of the configurations contained in `A` which specifies what sheets to use for the DB (e.g., `pv_msa_red` for a DB using only PV MSA lexicon, the `Order-v4` order sheet, and a **red**uced morph file for more efficient debugging). It also specifies what the name of the output DB file will be.
- Note that only one of `B` or `C` can be set. If `C` is set, then:
  - `D`: used for lexicon sheets (usually contain `LEX` in their names)
    - `D1`: name of spreadsheet
    - `D2 [...]`: name of sheets contained in `D1` to download in csv format
  - `E`: used for specs sheets (`ORDER`, `MORPH`, `HEADER`, ...)
    - `E1`: name of spreadsheet (use `header-morph-order-sheets` for now)
    - `E2 [...]`: name of sheets contained in `D1` to download in csv format

### Compile Database

The below command compiles the ALMOR-style database.

    python db_maker.py -config_file A -config_name B -camel_tools C [-output_dir D] [-run_profiling]

- `A`: same as above.
- `B`: same as above.
- `C`: directory containing the CAMeL Tools modules (clone this from [this](https://github.com/christios/camel_tools) fork).
- `D`: directory to save the generated DBs in. By default, `./db_iterations`

The following flag is optional:

- `-run_profiling`: to generate an execution time profile of the specific configuration.

### Generate Representative Lemmas List

The following script generates a list of representative lemmas to run the conjugation paradigms on (mainly for debugging). Unique lemmas are carefully chosen from distinct lemma classes based on stem, COND-T, COND-S, POS, and gloss information (and gender and number for nominals).

    python create_repr_lemmas_list.py -config_file A -config_name B -output_name C -pos_type D [-output_dir E]

To run the above command, the following flags are required:

- `A`: same as above.
- `B`: same as above.
- `C`: name of the output pickled dictionary. This is a pickle file (consider suffixing it with `.pkl` for consistency) containing lemmas and some info about them (`COND-S`, `COND-T`, `gen`, etc.).
- `D`: POS type of the lemmas we want a representative list of. Choices are `verbal` or `nominal`.
- `E`: path of the directory to output the file to. By default, `./conjugation/repr_lemmas` (if it does not exist, it will be automatically created).

### Generate Conjugation Tables

**Note:** to use the below code, a specific fork of CAMeL tools needs to be downloaded for debugging purposes. After downloading it from [here](https://github.com/christios/camel_tools/tree/master), extract the `camel_tools` package from the outer `camel_tools` build folder and place it in `./`. Alternatively, create a new virtual environment and manually install the fork there as a library.

    python generate_conj_table.py -paradigms A -repr_lemmas B -db C [-db_dir D] -pos_type E -dialect F -output_name G [-output_dir H] [-asp I] [-mod J]

To run the above command, the following flags are required:

- `A`: configuration file containing the paradigms from which to generate the tables. Paradigms are expressed as a set of signatures the syntax of which is as follows:

        <POS_TYPE>.<A><P><G><N>.<S><C><V><M>.<P0123><E012>
    where:

  - `<POS_TYPE>` can be either `VERB` or `NOM`.
  - `<A><P><G><N>.<S><C><V><M>` is as specified as in the [CAMeL POS](https://camel-guidelines.readthedocs.io/en/latest/morphology/) schema with the only exception that the plural value for `gen` is specified as `Q` instead of `P` (for ease of signature parsing)[^1].
  - The last part specifies the presence of clitics. For instance, if we want a 0-enclitic to be present, we add `.E0`. If we want both a 0- and a 1- enclitic, we add `.E01`, etc.
  
- `B`: file containing the representative lemmas list and their debug info.
- `C`: name of the DB file to use for generation.
- `D`: path of the directory in which `C` is. By default `./db_iterations`
- `E`: `verbal` or `nominal` (self-explanatory).
- `F`: `msa`, `glf`, or `egy`.
- `G`: name of the file to output the conjugation tables to. This file will be output by default to a directory called `conjugation/tables/`.
- `H`: path of the directory to which `G` will be output. By default `./conjugation/tables` (if it does not exist, it will be automatically created).
- `I`: `p`, `i`, or `c`. If we are generating verb conjugation tables, we need to specify what aspect we are generating for. Otherwise, do not use.
- `J`: `i`, `s`, or `j`. If we are generating `IV` verb conjugation tables, we need to specify what mood we want to generate for. Otherwise, do not use.

[^1]: The `Q` is reverted to a `P` when the signature is printed out again.
[^2]: Habash, Nizar et al. "Morphotactic Modeling in an Open-source Multi-dialectal Arabic Morphological Analyzer and Generator", SIGMORPHON. 2022
