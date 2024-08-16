# ls -al ./data/hcaptcha/cls/eval | wc -l
# ls -al ./data/hcaptcha/cls/data | wc -l

num_done=$(ls -al ./data/hcaptcha/cls/eval | wc -l)
num_total=$(ls -al ./data/hcaptcha/cls/data | wc -l)
percent=$(echo "scale=2; $num_done / $num_total * 100" | bc)
echo "finished $percent%"
