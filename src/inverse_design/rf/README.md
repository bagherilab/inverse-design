# ABC-SMC-DRF ARCADE Example

This script runs ABC-SMC-DRF (Approximate Bayesian Computation with Sequential Monte Carlo and Distributional Random Forest) on the ARCADE model.

## Usage

```bash
python3 arcade_example.py --config config.json --target target.json
```

## Configuration Files

### config.json
Contains all the configuration parameters for the ABC-SMC-DRF algorithm:

- `sobol_power`: Power for Sobol sequence sampling (default: 9)
- `n_group`: Number of groups for clustering (default: 3)
- `n_min_sample`: Minimum samples per group (default: 5)
- `radius`: Radius parameter (default: 10)
- `margin`: Margin parameter (default: 2)
- `hex_size`: Hexagon size (default: 30)
- `simplify_model`: Whether to simplify the model (default: true)
- `n_iterations`: Number of SMC iterations (default: 5)
- `rf_type`: Random forest type (default: "DRF")
- `n_trees`: Number of trees in the forest (default: 50)
- `min_samples_leaf`: Minimum samples per leaf (default: 5)
- `random_state`: Random seed (default: 42)
- `criterion`: Splitting criterion (default: "CART")
- `subsample_ratio`: Subsample ratio (default: 0.5)
- `correlation_threshold`: Correlation threshold for redundancy analysis (default: 0.8)
- `base_dir`: Base directory for input data (default: "../../../ARCADE_OUTPUT/ABC_SMC_RF_N1024_combined_grid_breast")

### target.json
Contains the target values for the optimization:

```json
{
    "doub_time": 45.5,
    "symmetry": 0.806,
    "colony_growth": 18.3,
    "doub_time_std": 13.79,
    "symmetry_std": 0.067
}
```

## Example

1. Create your configuration file:
```json
{
    "sobol_power": 8,
    "n_group": 4,
    "n_min_sample": 3,
    "radius": 12,
    "margin": 3,
    "n_iterations": 3,
    "n_trees": 100
}
```

2. Create your target file:
```json
{
    "doub_time": 50.0,
    "symmetry": 0.8,
    "colony_growth": 20.0
}
```

3. Run the script:
```bash
python3 arcade_example.py --config my_config.json --target my_targets.json
```

## Output

The script will:
1. Load the configuration and target values
2. Run the ABC-SMC-DRF algorithm
3. Generate plots and save results to the output directory
4. Print parameter estimation results

Results are saved in the output directory specified by the configuration. 

\
aws batch create-job-queue \
    --job-queue-name inverse-design-queue-2 \
    --state ENABLED \
    --priority 1 \
    --compute-environment-order order=1,computeEnvironment=ec2-ondemand \
    --region us-west-2

aws batch register-job-definition \
    --job-definition-name inverse-design \
    --type container \
    --container-properties '{
        "image": "<AWS_ACCOUNT_ID>.dkr.ecr.us-west-2.amazonaws.com/inverse-design-simulation:latest",
        "vcpus": 4,
        "memory": 8192,
        "command": ["python3", "src/rf/arcade_example.py", "Ref::args"]
    }' \
    --region us-west-2


aws batch submit-job \
    --job-name g2s4 \
    --job-queue inverse-design-queue-2 \
    --job-definition inverse-design-simulation \
    --parameters args="--config /app/configs/config_group2_sample4.json --target /app/configs/target.json"

aws batch create-compute-environment \
    --compute-environment-name arcade-compute-3 \
    --type MANAGED \
    --state ENABLED \
    --compute-resources '{
        "type": "EC2",
        "minvCpus": 0,
        "maxvCpus": 256,
        "desiredvCpus": 0,
        "instanceTypes": ["optimal"],
        "subnets": ["subnet-758d550d"],
        "securityGroupIds": ["sg-bf1c1393"],
        "instanceRole": "arn:aws:iam::<AWS_ACCOUNT_ID>:instance-profile/ec2_default"
    }' \
    --service-role arn:aws:iam::<AWS_ACCOUNT_ID>:role/ec2_default \
    --region us-west-2

    arn:aws:iam::<AWS_ACCOUNT_ID>:instance-profile/ec2_default

aws batch describe-job-definitions --job-definition-name inverse-design --region us-west-2 --query 'jobDefinitions[0].containerProperties.image'
<AWS_ACCOUNT_ID>.dkr.ecr.us-west-2.amazonaws.com/inverse-design-simulation:latest
aws batch describe-jobs --jobs fe5a047e-bfe3-4ee8-b544-4eef1985d263 --region us-west-2 --query 'jobs[0].{Status:status,StatusReason:statusReason,Attempts:attempts[0].statusReason}'
aws batch describe-compute-environments --compute-environments inverse-design --region us-west-2 --query 'computeEnvironments[0].computeResources.instanceRole'

aws batch describe-compute-environments --compute-environments ec2-ondemand --region us-west-2 --query 'computeEnvironments[0].computeResources.{MaxvCpus:maxvCpus,InstanceTypes:instanceTypes}'
aws batch describe-job-definitions --job-definition-name inverse-design --region us-west-2 --query 'jobDefinitions[0].containerProperties.{Vcpus:vcpus,Memory:memory}'
aws batch update-compute-environment \
    --compute-environment ec2-ondemand \
    --compute-resources desiredvCpus=8 \
    --region us-west-2


aws batch register-job-definition \
    --job-definition-name inverse-design-simulation-pchiu \
    --type container \
    --container-properties '{
        "image": "<AWS_ACCOUNT_ID>.dkr.ecr.us-west-2.amazonaws.com/pchiu/inverse-design-simulation:latest",
        "vcpus": 8,
        "memory": 4096,
        "jobRoleArn": "arn:aws:iam::<AWS_ACCOUNT_ID>:role/BatchJobRole",
        "command": ["sh", "-c", "cd /app/src/inverse_design && python3 rf/arcade_example.py --config configs/test.json --target configs/target.json"]
    }'

/usr/bin/docker run --rm \
    -e JAVA_OPTS="-Djava.awt.headless=true" \
    inverse-design-simulation \
    sh -c "cd /app/src/inverse_design && python3 rf/arcade_example.py --config ../../configs/test.json --target ../../configs/target.json"

docker run -e AWS_ACCESS_KEY_ID=$(aws configure get aws_access_key_id) \
           -e AWS_SECRET_ACCESS_KEY=$(aws configure get aws_secret_access_key) \
           -e AWS_DEFAULT_REGION=$(aws configure get region) \
           inverse-design-simulation \
    sh -c "cd /app/src/inverse_design && python3 rf/arcade_example.py --config ../../configs/test.json --target ../../configs/target.json"

aws batch register-job-definition \
    --job-definition-name inverse-design-simulation-mean-only-pchiu \
    --type container \
    --container-properties '{
        "image": "<AWS_ACCOUNT_ID>.dkr.ecr.us-west-2.amazonaws.com/pchiu/inverse-design-simulation-mean-only:latest",
        "vcpus": 128,
        "memory": 8192,
        "jobRoleArn": "arn:aws:iam::<AWS_ACCOUNT_ID>:role/BatchJobRole",
        "command": [
            "sh", "-c", 
            "cd /app/src/inverse_design && python3 rf/arcade_example.py --config ../../configs/config_combined_n1024_iter5.json --target ../../configs/target.json"
        ]
    }'

aws batch submit-job \
    --job-name inverse-design-test \
    --job-queue inverse-design-queue-pchiu \
    --job-definition inverse-design-simulation-pchiu

aws batch submit-job \
    --job-name breast-mean-only \
    --job-queue inverse-design-queue-pchiu \
    --job-definition inverse-design-simulation-mean-only-pchiu \
    --container-overrides '{
        "command": [
            "sh", "-c", 
            "cd /app/src/inverse_design && python3 rf/arcade_example.py --config ../../configs/config_combined_n1024_iter5.json --target ../../configs/target.json"
        ]
    }'

aws logs get-log-events --log-group-name /aws/batch/job --log-stream-name inverse-design-simulation-pchiu/default/da4f5fe8b9de4feb9b1ca31690fe1a08
