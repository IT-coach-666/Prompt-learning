需要安装的 packages：
  1. transformers
  2. seq2seq
  3. pandas
  4. tensorboardX

训练：
  python train.py
  默认使用 conll2003 数据，处理后的数据路径为：./data/train.csv 和 ./data/dev.csv

测试：
  python inference.py
  默认使用 conll2003 测试数据，./data/conll2003/test.txt

训练和测试使用的模板：
  location：is a location entity .
  person：is a person entity .
  organization：is an organization entity .
  other：is an other entity .
  none：is not a named entity .