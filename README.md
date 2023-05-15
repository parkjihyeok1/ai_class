# ai_class
#1. Introduction
Momentum contrast for unsupervised visual representation learning (MoCo) 논문
https://github.com/facebookresearch/moco  
https://github.com/ppwwyyxx/moco.tensorflow 에 있는 코드 활용

# 모델에 대한 설명 및 디테일  
Momentum Contrast for Unsupervised Visual Representation Learning (MoCo)는 자가 지도 학습(self-supervised learning)을 사용하여 시각적 표현 학습을 수행하는 방법을 제안한 논문입니다.   
이 방법은 사전 훈련된 모델을 사용하여 입력 이미지를 잠재 공간으로 임베딩하고, 이 임베딩을 사용하여 이미지 간의 유사성을 측정합니다.  
MoCo의 핵심 아이디어는 양성 샘플과 음성 샘플을 사용하여 이미지 간의 유사성을 학습하는 것입니다. 
이를 위해 두 개의 신경망, Query 신경망과 Key 신경망을 사용합니다.  
먼저, 원본 이미지와 이를 변형한 어그먼트 이미지를 사용하여 Query 신경망과 Key 신경망에 입력합니다.  
Query 신경망은 이미지를 잠재 공간으로 임베딩하는 역할을 합니다.  
Key 신경망은 임베딩된 이미지의 표현을 저장하는 역할을 합니다.  
데이터 샘플링:  
학습 데이터셋에서 이미지 쌍을 샘플링합니다. 각 이미지는 원본 이미지와 동일한 이미지에 대한 강화된 변형(증강)이 포함된 두 가지 버전으로 제공됩니다.  
특성 추출:  
인코더와 모멘텀 인코더를 사용하여 샘플링된 이미지의 특성 벡터를 추출합니다.  
대조 손실 계산:  
추출된 특성 벡터를 사용하여 대조 손실을 계산합니다. 이는 현재 인코더의 특성 벡터와 모멘텀 인코더의 특성 벡터 간의 일관성을 측정하는데 사용됩니다.  
역전파 및 가중치 업데이트:  
계산된 대조 손실을 기반으로 역전파를 수행하고, 인코더의 가중치를 업데이트합니다.  
모멘텀 인코더는 일정한 간격으로 현재 인코더의 가중치로 업데이트됩니다.  
반복:  
위 단계를 반복하여 모델을 계속 학습시킵니다. 일반적으로 많은 수의 이미지 쌍을 사용하여 여러 번의 반복을 수행합니다.  
이러한 모델의 학습 과정을 통해 MoCo는 비지도 학습을 통해 이미지의 표현을 향상시킵니다. 이후, 훈련된 인코더는 다양한 컴퓨터 비전 작업에 전이학습을 적용하여 유용한 특성을 추출하는 데 사용될 수 있습니다.


# 슈도코드
```
#모멘텀 대조 (MoCo) 알고리즘의 수학적인 원리를 포함한 슈도코드  
#동적 모델의 인코더 초기화  
encoder = DynamicEncoder()  
#모멘텀 인코더 초기화  
momentum_encoder = MomentumEncoder()  
#동적 모델과 모멘텀 인코더를 사용하여 사전 훈련된 인코더 구축  
def build_pretrained_encoder(dataset)  
    for image in dataset  
        features = encoder(image)  # 동적 모델을 사용하여 특성 벡터 생성  
        momentum_features = momentum_encoder.get_features()  # 모멘텀 인코더의 특성 벡터 가져오기  
        momentum_encoder.update(features)  # 모멘텀 인코더 업데이트  
        # 대조 손실 계산  
        loss = -log(exp(features * momentum_features / τ) / sum(exp(features * momentum_features / τ)))  
        # 동적 모델 최적화  
        optimizer.step(loss)  
#데이터로부터 표현 학습을 위한 특성 벡터 생성  
def generate_features(image)  
    features = encoder(image)  
    return features  
#특성 벡터의 일관성 검사  
def check_consistency(features)  
    momentum_features = momentum_encoder.get_features()  
    # 대조 손실 계산  
    loss = -log(exp(features * momentum_features / τ) / sum(exp(features * momentum_features / τ)))  
    return loss  
#주어진 특성 벡터를 사용하여 선형 분류기를 훈련  
def train_linear_classifier(features, labels)  
    classifier = LinearClassifier()  
    classifier.train(features, labels)  
#예시 데이터셋으로 사전 훈련된 인코더 구축  
dataset = ExampleDataset()  
build_pretrained_encoder(dataset)  
#특정 이미지로부터 특성 벡터 생성  
image = ExampleImage()  
features = generate_features(image)  
#특성 벡터의 일관성 검사  
loss = check_consistency(features)  
#특성 벡터를 사용하여 선형 분류기 훈련  
labels = ExampleLabels()  
train_linear_classifier(features, labels)  
```
# 모델에 대한 설명 및 디테일  
Momentum Contrast for Unsupervised Visual Representation Learning (MoCo)는 자가 지도 학습(self-supervised learning)을 사용하여 시각적 표현 학습을 수행하는 방법을 제안한 논문입니다.   
이 방법은 사전 훈련된 모델을 사용하여 입력 이미지를 잠재 공간으로 임베딩하고, 이 임베딩을 사용하여 이미지 간의 유사성을 측정합니다.  
MoCo의 핵심 아이디어는 양성 샘플과 음성 샘플을 사용하여 이미지 간의 유사성을 학습하는 것입니다. 
이를 위해 두 개의 신경망, Query 신경망과 Key 신경망을 사용합니다.  
먼저, 원본 이미지와 이를 변형한 어그먼트 이미지를 사용하여 Query 신경망과 Key 신경망에 입력합니다.  
Query 신경망은 이미지를 잠재 공간으로 임베딩하는 역할을 합니다.  
Key 신경망은 임베딩된 이미지의 표현을 저장하는 역할을 합니다.  
데이터 샘플링:  
학습 데이터셋에서 이미지 쌍을 샘플링합니다. 각 이미지는 원본 이미지와 동일한 이미지에 대한 강화된 변형(증강)이 포함된 두 가지 버전으로 제공됩니다.  
특성 추출:  
인코더와 모멘텀 인코더를 사용하여 샘플링된 이미지의 특성 벡터를 추출합니다.  
대조 손실 계산:  
추출된 특성 벡터를 사용하여 대조 손실을 계산합니다. 이는 현재 인코더의 특성 벡터와 모멘텀 인코더의 특성 벡터 간의 일관성을 측정하는데 사용됩니다.  
역전파 및 가중치 업데이트:  
계산된 대조 손실을 기반으로 역전파를 수행하고, 인코더의 가중치를 업데이트합니다.  
모멘텀 인코더는 일정한 간격으로 현재 인코더의 가중치로 업데이트됩니다.  
반복:  
위 단계를 반복하여 모델을 계속 학습시킵니다. 일반적으로 많은 수의 이미지 쌍을 사용하여 여러 번의 반복을 수행합니다.  
이러한 모델의 학습 과정을 통해 MoCo는 비지도 학습을 통해 이미지의 표현을 향상시킵니다. 이후, 훈련된 인코더는 다양한 컴퓨터 비전 작업에 전이학습을 적용하여 유용한 특성을 추출하는 데 사용될 수 있습니다.  


# Dependencies
TensorFlow 1.14 or 1.15, built with XLA support  
Tensorpack ≥ 0.10.1  
Horovod ≥ 0.19 built with Gloo & NCCL support  
TensorFlow zmq_ops  
OpenCV  
the taskset command (from the util-linux package)  
# Run
Unsupervised Training:  
To run MoCo pre-training on a machine with 8 GPUs, use:  
horovodrun -np 8 --output-filename moco.log python main_moco.py --data /path/to/imagenet  
Add --v2 to train MoCov2, which uses an extra MLP layer, extra augmentations, and cosine LR schedule.  
Linear Classification:  
To train a linear classifier using the pre-trained features, run:  
./main_lincls.py --load /path/to/pretrained/checkpoint --data /path/to/imagenet  
KNN Evaluation:  
Instead of Linear Classification, a cheap but rough evaluation is to perform a feature-space kNN using the training set:  
horovodrun -np 8 ./eval_knn.py --load /path/to/checkpoint --data /path/to/imagenet --top-k 200  
메인코드는 data.py , eval_knn.py , main_moco.py , resent.py , serve-data.py , main_lincls.py 를 실행해야합니다.  
