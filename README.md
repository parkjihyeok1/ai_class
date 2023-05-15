# ai_class
1. Introduction
Momentum contrast for unsupervised visual representation learning (MoCo) 논문
https://github.com/facebookresearch/moco 깃허브 주소
# 슈도코드
모멘텀 대조 (MoCo) 알고리즘의 수학적인 원리를 포함한 슈도코드
# 동적 모델의 인코더 초기화
encoder = DynamicEncoder()
# 모멘텀 인코더 초기화
momentum_encoder = MomentumEncoder()
# 동적 모델과 모멘텀 인코더를 사용하여 사전 훈련된 인코더 구축
def build_pretrained_encoder(dataset):
    for image in dataset:
        features = encoder(image)  # 동적 모델을 사용하여 특성 벡터 생성
        momentum_features = momentum_encoder.get_features()  # 모멘텀 인코더의 특성 벡터 가져오기
        momentum_encoder.update(features)  # 모멘텀 인코더 업데이트
        # 대조 손실 계산
        loss = -log(exp(features * momentum_features / τ) / sum(exp(features * momentum_features / τ)))
        # 동적 모델 최적화
        optimizer.step(loss)
# 데이터로부터 표현 학습을 위한 특성 벡터 생성
def generate_features(image):
    features = encoder(image)
    return features
# 특성 벡터의 일관성 검사
def check_consistency(features):
    momentum_features = momentum_encoder.get_features()
    # 대조 손실 계산
    loss = -log(exp(features * momentum_features / τ) / sum(exp(features * momentum_features / τ)))
    return loss
# 주어진 특성 벡터를 사용하여 선형 분류기를 훈련
def train_linear_classifier(features, labels):
    classifier = LinearClassifier()
    classifier.train(features, labels)
# 예시 데이터셋으로 사전 훈련된 인코더 구축
dataset = ExampleDataset()
build_pretrained_encoder(dataset)
# 특정 이미지로부터 특성 벡터 생성
image = ExampleImage()
features = generate_features(image)
# 특성 벡터의 일관성 검사
loss = check_consistency(features)
# 특성 벡터를 사용하여 선형 분류기 훈련
labels = ExampleLabels()
train_linear_classifier(features, labels)

2. 모델에 대한 설명 및 디테일
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

3.Requirements
Python 3.6 이상, Install PyTorch and ImageNet dataset following the official PyTorch ImageNet training code.
This repo aims to be minimal modifications on that code. Check the modifications by:
diff main_moco.py <(curl https://raw.githubusercontent.com/pytorch/examples/master/imagenet/main.py)
diff main_lincls.py <(curl https://raw.githubusercontent.com/pytorch/examples/master/imagenet/main.py)

python main_moco.py \
  -a resnet50 \
  --lr 0.03 \
  --batch-size 256 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  [your imagenet-folder with train and val folders]
  
This script uses all the default hyper-parameters as described in the MoCo v1 paper. To run MoCo v2, set --mlp --moco-t 0.2 --aug-plus --cos.
Note: for 4-gpu training, we recommend following the linear lr scaling recipe: --lr 0.015 --batch-size 128 with 4 gpus. We got similar results using this setting.


