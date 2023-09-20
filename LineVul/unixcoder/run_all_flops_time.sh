set -e

# for v in "" "_noflowgnn" "_nocuda" "_noflowgnn_nocuda"
for v in "_noflowgnn" "_noflowgnn_nocuda"
do
    # bash "scripts/rq1_eval_flops${v}.sh" MSR-LineVul
    bash "scripts/rq1_eval_time${v}.sh" MSR-LineVul
done

bash "scripts/rq1_eval_flops_noflowgnn.sh" MSR-LineVul
