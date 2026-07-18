1. 데이터의 목적

이 데이터는 미국 아이오와주 에임스(Ames) 지역에서 판매된 주택 정보를 담고 있습니다.

주택의 위치, 면적, 품질, 건축연도, 방 개수, 차고, 지하실, 거래조건 등을 이용해 최종 판매가격인 SalePrice를 분석하는 데이터입니다.

문제 유형: 회귀
분석 대상: 주택 판매가격
정답 변수: SalePrice
식별번호: Id

① 집의 기본 유형
변수	뜻
MSSubClass	주택 종류를 나타내는 코드
BldgType	단독주택, 복층주택, 타운하우스 등
HouseStyle	1층, 2층, 1.5층 등
MSZoning	저밀도·중밀도 등 토지 용도지역

② 위치와 토지
변수	뜻
Neighborhood	주택이 위치한 지역
LotArea	전체 토지 면적
LotFrontage	도로와 접한 길이
LotShape	토지 모양
LandContour	토지의 평탄도
Condition1	철도, 간선도로, 공원 등 주변 환경

③ 집의 크기
변수	뜻
GrLivArea	실제로 생활하는 지상 면적
1stFlrSF	1층 면적
2ndFlrSF	2층 면적
TotalBsmtSF	전체 지하실 면적
TotRmsAbvGrd	전체 방 개수
BedroomAbvGr	침실 개수

④ 집의 품질과 상태
변수	뜻
OverallQual	전체 자재와 마감 품질, 1~10점
OverallCond	집의 현재 상태, 1~10점
ExterQual	외장재 품질
KitchenQual	주방 품질
HeatingQC	난방 품질
BsmtQual	지하실 품질

⑤ 건축 및 리모델링 연도
변수	뜻
YearBuilt	처음 지어진 연도
YearRemodAdd	리모델링한 연도
GarageYrBlt	차고를 지은 연도
YrSold	판매된 연도
MoSold	판매된 월

⑥ 편의시설
변수	뜻
FullBath	전체 욕실 수
HalfBath	반쪽 욕실 수
GarageCars	차고에 주차할 수 있는 차량 수
GarageArea	차고 면적
Fireplaces	벽난로 수
PoolArea	수영장 면적
OpenPorchSF	개방형 현관 면적
WoodDeckSF	목재 데크 면적

⑦ 판매 조건
변수	뜻
SaleType	일반매매, 신축판매 등 판매 유형
SaleCondition	정상거래, 급매, 가족 간 거래 등
MoSold	판매된 월
YrSold	판매된 연도