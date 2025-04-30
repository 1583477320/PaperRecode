import openxlab
from openxlab.dataset import info
from openxlab.dataset import get
from openxlab.dataset import download

openxlab.login(ak='b7ke6mqblqgkr0pgdykz', sk='qv3blqam2wr8ykxg7w0d1vljog9ej0oopndbm1kz') # 进行登录，输入对应的AK/SK，可在个人中心添加AK/SK

info(dataset_repo='OpenDataLab/MultiMNIST') #数据集信息查看


# get(dataset_repo='OpenDataLab/MultiMNIST', target_path=r'C:\Users\15834\PycharmProjects\paper_recode') # 数据集下载
#
#
download(dataset_repo='OpenDataLab/MultiMNIST',source_path='/README.md', target_path=r'C:\Users\15834\PycharmProjects\paper_recode') #数据集文件下载