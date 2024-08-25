# coding:utf-8
'''
code for id
'''
from PIL import ImageFont,Image,ImageDraw
from utils.idcard_entity import *
# pip install opencv-python
import cv2
import numpy as np

def paste_photo(img, img_back, zoom_size, center):
    img = cv2.resize(img, zoom_size)
    rows, cols, channels = img.shape
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    diff = [5, 30, 30]
    gb = hsv[0, 0]
    lower_blue = np.array(gb - diff)
    upper_blue = np.array(gb + diff)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    for i in range(rows):
        for j in range(cols):
            if mask[i, j] == 0:
                img_back[center[0] + i, center[1] + j] = img[i, j]
    return img_back

def generator(idcard_info, name_font,other_font,bdate_font,
              id_font,idcard_empty_path,idcard_photo_path,save_dir,bk_path=None):
    # 得到身份证信息
    name = idcard_info.get_name()
    gender = idcard_info.get_gender()
    nation = idcard_info.get_nation()
    year = str(idcard_info.get_birth_year())
    mon = str(idcard_info.get_birth_month())
    day = str(idcard_info.get_birth_day())
    org = idcard_info.get_province_name() + idcard_info.get_city_name() + '公安局'
    valid_start = year+'.'
    valid_start = valid_start+'0'+mon if len(mon) == 1 else valid_start+ mon
    valid_start = valid_start+'.0'+day if len(day) == 1 else valid_start + "."+day
    valid_end = str(int(year)+20)+valid_start[4:]
    valid_range = valid_start+'-'+valid_end
    addr = idcard_info.get_addr()
    idn = idcard_info.get_id()

    im = Image.open(idcard_empty_path)
    imarr = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)

    # 在写字之前，换背景
    if bk_path is not None:
        bk = cv2.imread(bk_path)
        lower_blue = np.array([0, 0, 0])
        upper_blue = np.array([5, 5, 5])
        bg_mask_bin = cv2.inRange(imarr, lower_blue, upper_blue)

        bg_mask = cv2.cvtColor(bg_mask_bin, cv2.COLOR_GRAY2BGR)
        bk = cv2.resize(bk, (bg_mask.shape[1], bg_mask.shape[0]))
        bg_add = cv2.bitwise_and(bk, bg_mask)
        imarr = cv2.add(imarr, bg_add)

    im = Image.fromarray(cv2.cvtColor(imarr, cv2.COLOR_BGR2RGB))
    avatar = Image.open(idcard_photo_path)  # 500x670

    draw = ImageDraw.Draw(im)
    draw.text((630, 690), name, fill=(0, 0, 0), font=name_font)
    draw.text((630, 840), gender, fill=(0, 0, 0), font=other_font)
    draw.text((1030, 840), nation, fill=(0, 0, 0), font=other_font)
    draw.text((630, 980), year, fill=(0, 0, 0), font=bdate_font)
    draw.text((950, 980), mon, fill=(0, 0, 0), font=bdate_font)
    draw.text((1150, 980), day, fill=(0, 0, 0), font=bdate_font)
    start = 0
    loc = 1120
    while start + 11 < len(addr):
        draw.text((630, loc), addr[start:start + 11], fill=(0, 0, 0), font=other_font)
        start += 11
        loc += 100
    draw.text((630, loc), addr[start:], fill=(0, 0, 0), font=other_font)
    draw.text((870, 1475), idn, fill=(0, 0, 0), font=id_font)
    #背面信息
    draw.text((1050, 2750), org, fill=(0, 0, 0), font=other_font)
    draw.text((1050, 2895), valid_range, fill=(0, 0, 0), font=other_font)
    avatar = cv2.cvtColor(np.asarray(avatar), cv2.COLOR_RGBA2BGRA)
    im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGBA2BGRA)
    im = paste_photo(avatar, im, (500, 670), (690, 1500))
    cv2.imwrite(save_dir+"idcard_full.jpg", im)
    # 分割
    split_image_vertically(save_dir+"idcard_full.jpg",save_dir)


def split_image_vertically(image_path:str, save_dir:str):
    # 打开图片
    image = Image.open(image_path)

    # 获取图片的宽度和高度
    width, height = image.size

    # 计算中间位置
    middle_height = height // 2

    # 裁剪上半部分
    top_image = image.crop((0, 0, width, middle_height))

    # 裁剪下半部分
    bottom_image = image.crop((0, middle_height, width, height))

    # 保存裁剪后的图片
    top_image.save(save_dir+'idcard_back.jpg')
    bottom_image.save(save_dir+'idcard_front.jpg')



if __name__=="__main__":
    name_font = ImageFont.truetype('resource/font/hei.ttf', 72)
    other_font = ImageFont.truetype('resource/font/hei.ttf', 60)
    bdate_font = ImageFont.truetype('resource/font/fzhei.ttf', 60)
    id_font = ImageFont.truetype('resource/font/ocrb10bt.ttf', 72)

    random_sex = random.randint(0, 1)  # 随机生成男(1)或女(0)
    areaCode = "500101"
    id = generate_id(areaCode, random_sex)
    provinceName="重庆市"
    cityName = "万州区"
    addr = "重庆市万州区园丁路41号"
    name = "张三"
    nation = "土家"
    idcard = IDCard(name,nation,addr,id,provinceName,cityName)
    idcard_empty_path = "resource/bkimg/idcard_empty.png"
    idcard_photo_path = "resource/bkimg/idcard_avatar.png"
    bk_path = "resource/bkimg/black.jpg"
    idcard_save_dir = "res/"
    generator(idcard, name_font, other_font, bdate_font, id_font,idcard_empty_path,
              idcard_photo_path,idcard_save_dir,bk_path)