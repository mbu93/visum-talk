import json
import os
from functools import wraps
from textwrap import dedent
from typing import List

def trial_template(name, image, replicas, model, data, outdir, repo):
    args = ["python",
            "main.py", 
            "preprocess", 
            "--out",
            "." ,
            "&&",
            "python", 
            "main.py", 
            "tune",
            "--out",
            ".",
            "{{- with .HyperParameters}}",
            "{{- range .}}",
            "{{.Name}}={{.Value}}",
            "{{- end}} {{- end}}"]

    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": "{{.Trial}}",
            "namespace": "{{.NameSpace}}"
        },
        "spec": {
            "template": {
                "spec": {
                    "nodeSelector": {
                        "accelarator": "nvidia"
                    },
                    "containers": [{
                        "name": name,
                        "image": image,
                        "imagePullPolicy": "Always",
                        "command": args,
                    }],
                    "restartPolicy": "OnFailure",
                }
            }
        }
    }


class KatibOp():
    is_created = False

    def __init__(self, image: str, name: str, output: str, repo: str, model: str, data: str):
        self.data = data
        self.model = model
        self.repo = repo
        self.output = output
        self.name = name
        self.image = image
        self.controller_img = "mbu93/katib-launcher:latest"
        self.with_objective("maximize", 0.95, "accuracy")
        self.with_algorithm("random")

        self.with_parameters(["model", "gamma", "neighbors"], ["categorical", "double", "int"], [
                             {"list": ["sklearn.svm.SVC", "sklearn.neighbors.KNeighborsClassifier", "sklearn.linear_model.LinearRegression"]},
                             {"min": "0.0001", "max": "0.01"}, {"min": "5", "max": "128"}])
        self.with_trial_template(
            name="pytorch",
            image=image,
            replicas=1,
            outdir=output,
            repo=repo
        )
        self.with_metrics_collector_spec("StdOut")
        self.create()
        self.is_created = True

    def rebuild(func):
        @wraps(func)
        def wrap(*args, **kwargs):
            self = args[0]
            func(*args, **kwargs)
            if self.is_created:
                self.create()
        return wrap

    @rebuild
    def with_objective(self, goal_type: str, goal: float, metric: str):
        self.objectiveConfig = {
            "type": goal_type,
            "goal": goal,
            "objectiveMetricName": metric,
        }
        return self

    @rebuild
    def with_algorithm(self, algorithm_type: str):
        self.algorithmConfig = {"algorithmName": algorithm_type}
        return self

    @rebuild
    def with_parameters(self, names: List[str], types: List[str], spaces: List[dict]):
        self.parameters = [{"name": "--" + x, "parameterType": y, "feasibleSpace": z}
                           for x, y, z in zip(names, types, spaces)]
        return self

    @rebuild
    def with_trial_template(self, image: str, name: str, replicas: int, outdir: str, repo: str):
        rawTemplate = trial_template(name=name, image=image, replicas=replicas, model=self.model,
                                     outdir=self.output, data=self.data, repo=self.repo)
        self.trialTemplate = {
            "goTemplate": {
                "rawTemplate": json.dumps(rawTemplate)
            }
        }
        return self

    @rebuild
    def with_metrics_collector_spec(self, kind: str, src: str = None):
        self.metricsCollectorSpec = {
            "collector": {
                "kind": kind
            }
        } if not src else {
            "collector": {
                "kind": kind,
                "source": {
                    "fileSystemPath": {
                        "path": src,
                        "kind": "File"
                    }
                }
            }
        }

        return self

    def create(self):
        self.op = dedent("""
        python /ml/launch_experiment.py \
        --name {} \
        --namespace {} \
        --maxTrialCount {} \
        --parallelTrialCount {} \
        --objectiveConfig '{}' \
        --algorithmConfig '{}' \
        --trialTemplate '{}' \
        --parameters '{}' \
        --metricsCollector '{}' \
        --experimentTimeoutMinutes '{}' \
        --deleteAfterDone {} \
        --outputFile results/params
        """.format(
            self.name,
            os.environ["KATIB_NS"] if "KATIB_NS" in os.environ else "mb18lasy",
            6,
            3,
            json.dumps(self.objectiveConfig),
            json.dumps(self.algorithmConfig),
            json.dumps(self.trialTemplate),
            json.dumps(self.parameters),
            json.dumps(self.metricsCollectorSpec),
            15,
            True,
        ))
