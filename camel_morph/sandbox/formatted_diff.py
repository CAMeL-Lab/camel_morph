import argparse
import os
import sys
import pandas as pd
from numpy import nan

COLUMNS_PROJECTS = dict(
    camel_morph=dict(
        columns=['LINE', 'PATTERN', 'ROOT', 'DEFINE', 'CLASS', 'LEMMA', 'FORM',
                 'BW', 'GLOSS', 'FEAT', 'COND-T', 'COND-S'],
        id='LINE'),
    maknuune=dict(
        columns=['ID', 'ROOT', 'ROOT_NTWS', 'LEMMA', 'FORM', 'CAPHI++',
                 'ANALYSIS', 'GLOSS', 'GLOSS_MSA', 'EXAMPLE_USAGE',
                 'NOTES', 'SOURCE', 'ANNOTATOR'],
        id='ID')
)

def _get_index2row(df):
    return {row[ID_INDEX]: tuple(row[:ID_INDEX] + row[ID_INDEX+1:])
            for row in df[COLUMNS].values.tolist()}

def _check_soundness_of_indexes(df):
    df_indexes = df[ID_COL].values.tolist()
    assert len(set(df_indexes)) == len(df_indexes)

def formatted_diff(select_for_compare, compare_with_selected):
    before = _get_index2row(select_for_compare)
    after = _get_index2row(compare_with_selected)

    after2before = {}
    for index_after, row_after in after.items():
        if index_after in before:
            row_before = before[index_after]
            # Row did not change
            if row_after == row_before:
                after2before[index_after] = 'no_edit'
            # An edit happened
            else:
                after2before[index_after] = [i for i, x in enumerate(row_after) if x != row_before[i]]
        # Row was added
        else:
            after2before[index_after] = 'new'
    # Row was deleted
    unused_indexes = set(before) - set(after)
    unused_rows = []
    for index_before in unused_indexes:
        row_ = list(before[index_before])
        row_.insert(ID_INDEX, index_before)
        row_.insert(DIFF_INDEX, 'deleted')
        unused_rows.append(row_)

    output = []
    for index_after, diffs in sorted(after2before.items(), key=lambda row: row[ID_INDEX]):
        if diffs in ['no_edit', 'new']:
            output_ = [index_after] + list(after[index_after]) + [diffs]
        else:
            output_ = [index_after]
            output_ += list(after[index_after])
            output_ += [' '.join(f'{COLUMNS_ESSENTIAL[i].lower()}:{col}'
                        for i, col in enumerate(before[index_after]) if i in diffs)]
        output.append(output_)
    
    output += unused_rows
    output = sorted(output, key=lambda row: row[ID_INDEX])
    output.insert(0, COLUMNS_OUTPUT)

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-select_for_compare", default='',
                        type=str, help="Path of the file to select for compare (left).")
    parser.add_argument("-compare_with_selected", default='',
                        type=str, help="Path of the file to compare with selected (right).")
    parser.add_argument("-output_path", default='',
                        type=str, help="Path of the files to output to.")
    parser.add_argument("-project", required=True,
                        type=str, help="Name of the project. Used to determine which columns to use.")
    parser.add_argument("-camel_tools", default='',
                        type=str, help="Path of the directory containing the camel_tools modules.")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    if args.camel_tools:
        camel_tools_dir = args.camel_tools
        sys.path.insert(0, camel_tools_dir)

    from camel_tools.utils.charmap import CharMapper

    ar2bw = CharMapper.builtin_mapper('ar2bw')
    bw2ar = CharMapper.builtin_mapper('bw2ar')

    COLUMNS = COLUMNS_PROJECTS[args.project]['columns']
    ID_COL = COLUMNS_PROJECTS[args.project]['id']
    COLUMNS_ESSENTIAL = [col for col in COLUMNS if col != ID_COL]
    COLUMNS_OUTPUT = COLUMNS + ['DIFF']
    ID_INDEX = COLUMNS.index(COLUMNS_PROJECTS[args.project]['id'])
    DIFF_INDEX = COLUMNS_OUTPUT.index('DIFF')

    select_for_compare = pd.read_csv(args.select_for_compare, sep=None)
    select_for_compare = select_for_compare.replace(nan, '')
    compare_with_selected = pd.read_csv(args.compare_with_selected, sep=None)
    compare_with_selected = compare_with_selected.replace(nan, '')
    for i, row in compare_with_selected.iterrows():
        if not row[ID_COL]:
            compare_with_selected.loc[i, ID_COL] = compare_with_selected.loc[i - 1, ID_COL] + 0.001

    _check_soundness_of_indexes(select_for_compare)
    _check_soundness_of_indexes(compare_with_selected)
    
    output = formatted_diff(select_for_compare, compare_with_selected)
    
    with open(args.output_path, 'w') as f:
        for i, row in enumerate(output):
            if i and type(row[ID_INDEX]) is float and row[ID_INDEX].is_integer():
                row[ID_INDEX] = int(row[ID_INDEX])
            print(*row, sep='\t', file=f)
        
