import csv
import json
import os
from itertools import islice
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.mrs.v20200910 import mrs_client, models


def main():
    secret_id = 'AKIDEH6fwwhANJxLTJsABvOaV8UZNvDEyRdY'
    secret_key = 'g5D4ey0lFUSAbrpWw5yavzmIWTAajbCX'
    """
    报告类型，目前支持12（检查报告），15（病理报告），28（出院报告），29（入院报告），210（门诊病历），212（手术记录），218（诊断证明），
    363（心电图），27（内窥镜检查），215（处方单），219（免疫接种证明），301（C14呼气试验）。如果不清楚报告类型，可以使用分类引擎，
    该字段传0（同时IsUsedClassify字段必须为True，否则无法输出结果）
    """
    text_type = 218
    """
    是否使用分类引擎，当不确定报告类型时，可以使用收费的报告分类引擎服务。若该字段为False，则Type字段不能为0，否则无法输出结果。 
    注意：当 IsUsedClassify 为True 时，表示使用收费的报告分类服务，将会产生额外的费用，具体收费标准参见 购买指南的产品价格。
    """
    is_used_classify = False
    text = '认知活动： 1.感知觉： 未见感觉增强、感觉减退或缺失、感觉倒错，存在言语性幻听，称凭空能听到声音讲话，否认错觉、视幻觉及感知综合障碍' \
           '。未见思维化声及功能性幻听。  2.思维和思维障碍:能建立有效交谈，对答切题，语音、语速、语量正常。未见思维散漫或破裂，未见思维奔逸或' \
           '思维迟缓，未见思维贫乏、思维云集及思维中断，未见语词新作及病理性象征性思维，未见逻辑倒错性思维及诡辩性思维。未引出关系妄想、' \
           '被害妄想、被窃妄想等妄想。未发现重复言语、刻板言语、模仿言语。无强迫观念和超价观念。 3.注意力:交谈中注意力能集中，未见明显注' \
           '意增强。无明显注意涣散、注意狭窄、注意固定及随境转移。 4.记忆力:患者远、近记忆力及即时记忆力粗查可。未发现顺行性遗忘、逆行性遗忘、' \
           '心因性遗忘、潜隐记忆、似曾相识症。 5.智能:患者一般智力可，一般常识、理解力、判断力、抽象概括能力与其文化程度相符。 6.自知力:无，' \
           '不认为自己有病，对疾病无认识分析力。 情感活动：情感稍低落，表情冷淡，眼神接触可，情感反应与内心体验及周围环境欠协调，未见明显病理' \
           '性激情发作及强制性哭笑。未见矛盾情感、情感倒错。  '

    with open(os.path.abspath('../../data/data_utf_8/integrate_data.csv'), 'r', encoding='utf-8-sig') as file:
        csv_reader = csv.reader(file)
        line_count = 0
        for line in islice(csv_reader, 1, None):
            # text = line[4]
            session(secret_id, secret_key, text, text_type, is_used_classify)
            line_count += 1
            print('accomplish API invoke')
            print(text)

            if line_count > 0:
                break



def session(secret_id, secret_key, text, text_type, is_used_classify):
    resp = 'None Output'
    try:
        cred = credential.Credential(secret_id, secret_key)
        http_profile = HttpProfile()
        http_profile.endpoint = "mrs.tencentcloudapi.com"

        client_profile = ClientProfile()
        client_profile.httpProfile = http_profile
        client = mrs_client.MrsClient(cred, "ap-shanghai", client_profile)

        req = models.TextToObjectRequest()
        params = {
            "Text": text,
            "Type": text_type,
            "IsUsedClassify": is_used_classify
        }
        req.from_json_string(json.dumps(params))

        resp = client.TextToObject(req)
    except TencentCloudSDKException as err:
        print(err)

    return resp


if __name__ == '__main__':
    main()




