i=2

while [ $i -le 500 ]
do
  echo Number: $i
  python backtest.py $i
  ((i++))
done