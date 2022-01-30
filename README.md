# CAMeL Morph

This repository contains code meant to build an ALMOR-style database from a set of morphological specification and lexicon spreadsheets. To generate a database and paradigm-specific conjugation tables (to test for soundness of rules), follow the below instructions[^1].

[^1]: These can be generated in one shot using the `make all` command thanks to the included `Makefile`.

## Compile Database

The below command compiles the ALMOR-style databse.

    python db_maker.py -specs_sheets A -config_file B -config_name C [-profiling]

- `A`: spreadsheet containing morph, order, and lexicon sheets (e.g., `CamelMorphDB-v1.0.xlsx`). This should be donwloaded from Google Sheets and placed in the same directory as `db_maker.py`. Latest iteration found [here](https://docs.google.com/spreadsheets/d/1CGjvOLHYL1mN-84t53Xfe5Vggk3aUWKv/edit#gid=116591440).
- `B`: configuration file contained in the same directory as `db_maker.py` (already included in repo, i.e., `config.json`) which contains different configurations to run the DB on (previsouly the `PARAMS` sheet in the specs spreadsheet). There are precompiled configurations in `config.json`, but new ones could be easily added on demand.
- `C`: specifies what elements to build the DB for (e.g., `pv_msa_order-v4_red` for a DB using only PV MSA verbs for the lexicon, using the `Order-v4` order sheet, and a *reduced* morph file for more efficient debugging).

The following flags are optional:

- `-profiling`: to generate an execution time profile of the specific configuration.

## Generate Representative Lemmas List

    python create_repr_lemmas_list.py -specs_sheets A -config_file B -config_name C -cmplx_morph D -output_name E -pos_type F

To run the above command, the following flags are required:

- `A`: same as above.
- `B`: same as above.
- `C`: same as above.
- `D`: complex morph sequence to generate the representative lemmas list from (e.g., `[STEM-PV]`).
- `E`: path of the file to output the list to. This is a `csv` file containing lemmas and some info about them (`COND-S`, `COND-T`, `gen`, etc.).
- `F`: `verbal` or `nominal` (self-explanatory).

## Generate Conjugation Tables

**Note:** to use the below code, a specific fork of CAMeL tools needs to be downloaded for debugging purposes. After downloading it from [here](https://github.com/christios/camel_tools/tree/master), extract the `camel_tools` package from the outer `camel_tools` build folder and place it in the directory where `generate_conj_table.py` is sitting. Alternatively, create a new virtual environment, and manually install the fork there as a library.

    python generate_conj_table.py -paradigms A -repr_lemmas B -db C -pos_type D -dialect E -output_name F [-asp G] [-mod H]

To run the above command, the following flags are required:

- `A`: configuration file containing the paradigms from which to generate the tables. Paradigms are expressed as a set of signatures the syntax of which is as follows: `<POS_TYPE>.<A><P><G><N>.<S><C><V><M>.<P0123><E012>` where:

  - `<POS_TYPE>` can be either `VERB` or `NOM`.
  - `<A><P><G><N>.<S><C><V><M>` is as specified as in the [CAMeL POS](https://camel-guidelines.readthedocs.io/en/latest/morphology/) schema with the only exception that the plural value for `gen` is specified as `Q` instead of `P` (for ease of signature parsing)[^2].
  - The last part specifies the presence of clitics. For instance, if we want a 0-enclitic to be present, we add `.E0`. If we want both a 0- and a 1- enclitic, we add `.E01`, etc.
  
[^2]: The `Q` is reverted to a `P` when the signature is printed out again.
  
- `B`: file containing the representative lemmas list and their debug info.
- `C`: path of the DB file to use for generation.
- `D`: `verbal` or `nominal` (self-explanatory).
- `E`: `msa`, `glf`, or `egy`.
- `F`: path of the file to output the conjugation tables to.

The following flags are necessary depending on what is being generated:

- `-asp G`: `p`, `i`, or `c`. If we are generating verb conjugation tables, we need to specify what aspect we are generating for.
- `-mod H`: `i`, `s`, or `j`. If we are generating `IV` verb conjugation tables, we need to specify what mood we want to generate for.
