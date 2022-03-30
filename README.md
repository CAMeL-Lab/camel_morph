# CAMeL Morph

This repository contains code meant to build an ALMOR-style database from a set of morphological specification and lexicon spreadsheets. To generate a database and paradigm-specific conjugation tables (to test for soundness of rules), follow the below instructions. Note that the way all the below scripts are used is described in detail in the adjoining `Makefile` which can be used to reproduce all our experiments by running `make all`.

## Downloading Files

To download files from Google Drive, first, follow the instructions in the first 3:50 minutes of [this](https://www.youtube.com/watch?v=bu5wXjz2KvU) video to get API access to the Google Drive interface. Then, run the following command to download selected/individual sheets (not spreadsheets) in csv format from the following cloud [directory](https://drive.google.com/drive/folders/1yRq5PZ7rwQKzGCIIcoVPvgbTLHrkkxpE).

    python download_sheets.py [-save_dir A] [-config_file B -config_name C] [-lex D1 D2 [...] [-lex ...] -specs E1 E2 [...] [-specs ...]]

- `A`: path of the directory to save all the downloaded files in. It is `./data` by default (if it does not exist, it will be automatically created).
- `B`: path of the configuration file (already included in repo, i.e., `./config.json`) which contains different configurations to run the DB on (previsouly the `PARAMS` sheet in the specs spreadsheet). Some pre-compiled configurations already exist in `config.json`, but new ones could be easily added on demand.
- `C`: configuration name of one of the configurations contained in `A` which specifies what sheets to use for the DB (e.g., `pv_msa_order-v4_red` for a DB using only PV MSA lexicon, the `Order-v4` order sheet, and a **red**uced morph file for more efficient debugging). It also specifies what the name of the output DB file will be.
- Note that only one of `B` or `C` can be set. If `C` is set, then:
  - `D`: used for lexicon sheets (usually contain `LEX` in their names)
    - `D1`: name of spreadsheet
    - `D2 [...]`: name of sheets contained in `D1` to download in csv format
  - `E`: used for specs sheets (`ORDER`, `MORPH`, `HEADER`, ...)
    - `E1`: name of spreadsheet (use `header-morph-order-sheets` for now)
    - `E2 [...]`: name of sheets contained in `D1` to download in csv format

## Compile Database

The below command compiles the ALMOR-style database.

    python db_maker.py -config_file A -config_name B [-output_dir C] [-run_profiling]

- `A`: same as above.
- `B`: same as above.
- `C`: directory to save the generated DBs in. By default, `./db_iterations`

The following flag is optional:

- `-run_profiling`: to generate an execution time profile of the specific configuration.

## Generate Representative Lemmas List

The following script generates a list of representative lemmas to run the conjugation paradigms on (mainly for debugging). Unique lemmas are carefully chosen from distinct lemma classes based on stem, COND-T, COND-S, POS, and gloss information (and gender and number for nominals).

    python create_repr_lemmas_list.py -config_file A -config_name B -output_name C -pos_type D [-output_dir E]

To run the above command, the following flags are required:

- `A`: same as above.
- `B`: same as above.
- `C`: name of the output pickled dictionary. This is a pickle file (consider suffixing it with `.pkl` for consistency) containing lemmas and some info about them (`COND-S`, `COND-T`, `gen`, etc.).
- `D`: POS type of the lemmas we want a representative list of. Choices are `verbal` or `nominal`.
- `E`: path of the directory to output the file to. By default, `./conjugation/repr_lemmas` (if it does not exist, it will be automatically created).

## Generate Conjugation Tables

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
