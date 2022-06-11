import sys

sys.path.insert(
    0, "/Users/chriscay/Library/Mobile Documents/com~apple~CloudDocs/NYUAD/camel_tools")

from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer
from camel_tools.utils.charmap import CharMapper
ar2bw = CharMapper.builtin_mapper('ar2bw')
bw2ar = CharMapper.builtin_mapper('bw2ar')

db = MorphologyDB(
    '/Users/chriscay/Library/Mobile Documents/com~apple~CloudDocs/NYUAD/camelMorph/db_iterations_local/XYZ_egy_all_v1.0.db')
# db = MorphologyDB("sandbox/calima-msa-s31_0.4.2.utf8.db")
# db = MorphologyDB("eval/calima-egy-c044_0.2.0.utf8.db")
# analyzer = Analyzer(db, backoff='SMART')

analyzer = Analyzer(db)

# analyses = analyzer.analyze('سيقرؤون')
analyses = analyzer.analyze('ابلغتموها')
pass