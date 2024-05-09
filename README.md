

This repo includes code to experiment with distributed model training for a ML system; specifically, data loading/ingestion, training. Future steps would involve the complete e2e suite: model serving, monitoring. 

The data served to train an image classifier. 

## Set up
* kubectl
* minikube
* docker


The kubeflow training operator was used to assist in distributed training: 

```
kubectl apply -k "github.com/kubeflow/training-operator/manifests/overlays/standalone?ref=v1.7.0"
```
In order to reference a locally built docker image inside minikube when running the cluster:

```
eval $(minikube -p minikube docker-env)
```

Then we can build the image:

```
docker build -f Dockerfile -t kubeflow/multi-worker-strategy:v0.1 .
```

To have a saved model inside the pod, a PVC is used: 

```
kubectl create -f multi-worker-pvc.yaml
```

The tensorflow job that runs the script: 

```
kubectl create -f multi-worker-tfjob.yaml
```