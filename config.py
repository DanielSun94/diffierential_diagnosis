import os
import re


device = 'cuda:7'
cache_dir = '/data/sunzhoujian/hugginface/'
diagnosis_map = {'双相': 0, '抑郁': 1, '焦虑障碍': 2}
vocab_size_lda = 3000
topic_number_lda = 10
vocab_size_ntm = 3000
topic_number_ntm = 4
hidden_size_ntm = 128


cn_CLS_token = '[CLS]'
parse_list = [
    ['姓名', re.compile('\u59d3[\s\*\u0020]*\u540d[\uff1a:\n\t\s]+')],
    ['性别', re.compile('\u6027[\*\s]*\u522b[\uff1a:\n\t\s]+')],
    ['年龄', re.compile('\u5e74[\*\s]*\u9f84[\uff1a:\n\t\s]+'),
     re.compile('\u51fa[\*\s]*\u751f[\*\s]*\u65e5[\*\s]*\u671f[\uff1a:\n\t\s]+')],
    ['民族', re.compile('\u6c11[\*\s]*\u65cf[\uff1a:\n\t\s]+')],
    ['职业', re.compile('\u804c[\*\s]*\u4e1a[\uff1a:\n\t\s]+')],
    ['出生地', re.compile('\u51fa[\*\s]*\u751f[\*\s]*\u5730[\uff1a:\n\t\s]+')],
    ['婚姻', re.compile('\u5a5a[\*\s]*\u59fb[\uff1a:\n\t\s]+'),
     re.compile('\u5a5a[\*\s]*\u80b2[\*\s]*\u53f2[\uff1a:\n\t\s]+')],
    ['宗教', re.compile('\u5b97[\*\s]*\u6559[\uff1a:\n\t\s]+')],
    ['联系方式', re.compile('\u8054[\*\s]*\u7cfb[\*\s]*\u65b9[\*\s]*\u5f0f[\uff1a:\n\t\s]+')],
    ['家庭地址', re.compile('\u5bb6[\*\s]*\u5ead[\*\s]*\u5730[\*\s]*\u5740[\uff1a:\n\t\s]+')],
    ['入院时间', re.compile('\u5165[\*\s]*\u9662[\*\s]*\u65f6[\*\s]*\u95f4[\uff1a:\n\t\s]+'),
     re.compile('\u5165[\*\s]*\u9662[\*\s]*\u65e5[\*\s]*\u671f[\uff1a:\n\t\s]+')],
    ['记录时间', re.compile('\u8bb0[\*\s]*\u5f55[\*\s]*\u65f6[\*\s]*\u95f4[\uff1a:\n\t\s]+'),
     re.compile('\u8bb0[\*\s]*\u5f55[\*\s]*\u65e5[\*\s]*\u671f[\uff1a:\n\t\s]+')],
    ['身份证号', re.compile('\u8eab[\*\s]*\u4efd[\*\s]*\u8bc1[\*\s]*\u53f7[\uff1a:\n\t\s]+')],
    ['病史陈述者', re.compile('\u75c5[\*\s]*\u53f2[\*\s]*\u9648[\*\s]*\u8ff0[\*\s]*\u8005[\uff1a:\n\t\s]+')],
    ['主诉', re.compile('\u4e3b[\*\s]*\u8bc9[\uff1a:\n\t\s]+')],
    ['现病史', re.compile('\u73b0[\*\s]*\u75c5[\*\s]*\u53f2[\uff1a:\n\t\s]+')],
    ['既往史', re.compile('\u65e2[\*\s]*\u5f80[\*\s]*\u53f2[\uff1a:\n\t\s]+')],
    ['个人史', re.compile('\u4e2a[\*\s]*\u4eba[\*\s]*\u53f2[\uff1a:\n\t\s]+')],
    ['婚育史', re.compile('\u5a5a[\*\s]*\u80b2[\*\s]*\u53f2[\uff1a:\n\t\s]+')],
    ['家族史', re.compile('\u5bb6[\*\s]*\u65cf[\*\s]*\u53f2[\uff1a:\n\t\s]+')],
    ['月经史', re.compile('\u6708[\*\s]*\u7ecf[\*\s]*\u53f2[\uff1a:\n\t\s]+')],
    ['体格检查', re.compile('\u4f53[\*\s]*\u683c[\*\s]*\u68c0[\*\s]*\u67e5')],
    ['精神检查', re.compile('\u7cbe[\*\s]*\u795e[\*\s]*\u68c0[\*\s]*\u67e5[\uff1a:\n\t\s]+')],
    ['谈话记录', re.compile('\u8c08[\*\s]*\u8bdd[\*\s]*\u8bb0[\*\s]*\u5f55[\uff1a:\n\t\s]+')],
    ['辅助检查', re.compile('\u8f85[\*\s]*\u52a9[\*\s]*\u68c0[\*\s]*\u67e5[\uff1a:\n\t\s]+')],
    ['诊断', re.compile('\u5165[\*\s]*\u9662[\*\s]*\u8bca[\*\s]*\u65ad[\uff1a:\n\t\s]+'),
     re.compile('\u8865[\*\s]*\u5145[\*\s]*\u8bca[\*\s]*\u65ad[\uff1a:\n\t\s]+'),
     re.compile('\u521d[\*\s]*\u6b65[\*\s]*\u8bca[\*\s]*\u65ad[\uff1a:\n\t\s]+'),
     re.compile('\u8bca[\*\s]*\u65ad[\uff1a:\n\t\s]+')],
    ['父母近亲婚姻', re.compile('\u7236[\*\s]*\u6bcd[\*\s]*\u8fd1[\*\s]*\u4eb2[\*\s]*\u5a5a[\*\s]*\u59fb[\uff1a:\n\t\s]+')],
    ['寄养', re.compile('\u521d\u6b65\u8bca\u65ad[\uff1a:\n\t\s]+')],
    ['家系谱图', re.compile('\u5bb6[\*\s]*\u7cfb[\*\s]*\u8c31[\*\s]*\u56fe[\uff1a:\n\t\s]+')],
    ['神经系统检查', re.compile('\u795e[\*\s]*\u7ecf[\*\s]*\u7cfb[\*\s]*\u7edf[\*\s]*\u68c0[\*\s]*\u67e5[\uff1a:\n\t\s]+'),
     re.compile('\u795e[\*\s]*\u7ecf[\*\s]*\u4e13[\*\s]*\u79d1[\*\s]*\u60c5[\*\s]*\u51b5')],
    ['医师签名', re.compile('\u533b[\*\s]*\u5e08[\*\s]*\u7b7e[\*\s]*\u540d[\uff1a:\n\t\s]+')],
]

skip_word_set = {'，', '、', ':', '：', '。', '”', '“', ';'}

topic_model_admission_parse_list = ['主诉', '现病史', '既往史', '个人史', '家族史', '体格检查', '精神检查', '辅助检查',
                                    '神经系统检查', '性别', '年龄', '婚姻', '出生地', '职业']
topic_model_first_emr_parse_list = ['1、', '2、', '3、', '4、', '5、', '6、']
neural_network_admission_parse_list = ['现病史']
neural_network_first_emr_parse_list = []


integrate_file_name = os.path.abspath('./data/data_utf_8/integrate_data.csv')
emr_parse_file_path = os.path.abspath('./data/data_utf_8/病程记录解析序列.csv')

semi_structure_admission_path = os.path.abspath('./data/data_utf_8/半结构化入院记录.csv')
data_file_template = os.path.abspath('./data/origin_data/{}/{}.csv')
save_folder = os.path.abspath('./data/data_utf_8')
reorganize_first_emr_path = os.path.join(save_folder, 'first_emr_reorganize.csv')
tokenize_data_save_path = os.path.join(save_folder, 'tokenize_data.pkl')
word_count_path = os.path.join(save_folder, 'word_count.pkl')
