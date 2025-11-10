# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 19:35:05 2024

@author: Qiu
"""

'''
auto_control.py
该模块存放了本项目所需的自动控制函数

'''
import os
import cv2
import time
import pyautogui

def get_xy(img_mode_path):
    '''
    确定目标坐标
    Parameters
    ----------
    img_mode_path : String
        目标模版图路径

    Returns
    -------
    目标中心坐标组成的元组
    '''
    # 截取当前屏幕内容并保存到指定路径
    pyautogui.screenshot().save(r'process_image/screenshot.png')
    
    # 载入截图和模板图像
    img = cv2.imread(r'process_image/screenshot.png')
    img_goal = cv2.imread(img_mode_path)

    # 检查读取是否成功
    if img is None:
        raise FileNotFoundError("无法读取当前屏幕截图路径")
    if img_goal is None:
        raise FileNotFoundError(f"无法读取模板图路径: {img_mode_path}")

    # 转成灰度图像
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_goal_gray = cv2.cvtColor(img_goal, cv2.COLOR_BGR2GRAY)

    # 获取模板图的宽高
    height, width = img_goal_gray.shape[:2]

    # 匹配模板图
    result = cv2.matchTemplate(img_gray, img_goal_gray, cv2.TM_SQDIFF_NORMED)

    # 获取匹配结果区域的中心坐标
    upper_left = cv2.minMaxLoc(result)[2]
    lower_right = (upper_left[0] + width, upper_left[1] + height)
    avg = (int((upper_left[0] + lower_right[0]) / 2), int((upper_left[1] + lower_right[1]) / 2))

    return avg

def auto_click(img_mode_path, duration=0.5, clicks=1, button='left', interval=0.0):
    """
    自动移动并点击图标
    参数:
        img_mode_path: 模板图路径
        duration: 移动时间
        clicks: 点击次数
        button: 按键类型
        interval: 点击时间间隔
    """
    # 检查路径是否存在
    if not os.path.exists(img_mode_path):
        raise FileNotFoundError(f"无法找到模板图路径: {img_mode_path}")
    
    # 移动到目标位置并点击
    try:
        xy = get_xy(img_mode_path)
        pyautogui.moveTo(xy, duration=duration)
        pyautogui.click(clicks=clicks, button=button, interval=interval)
    except Exception as e:
        print(f"操作失败: {e}")

def register_image_by_location(ip, fp):
    '''
    自动注册图像
    参数：
        ip: Initial Point 初始点坐标
        fp: Final Point 最终点坐标
    '''
    pyautogui.hotkey('Ctrl', 'r')
    # 输入起始点和最终点的坐标
    pyautogui.moveTo(2002, 291, 0.5)
    pyautogui.click()
    pyautogui.press('backspace')
    pyautogui.write(str(ip[0]), interval=0.2)
    
    pyautogui.moveTo(2097, 297, 0.1)
    pyautogui.click()
    pyautogui.press('backspace')
    pyautogui.write(str(ip[1]), interval=0.2)
    
    pyautogui.moveTo(2009, 362, 0.1)
    pyautogui.click()
    pyautogui.press('backspace')
    pyautogui.write(str(fp[0]), interval=0.2)

    pyautogui.moveTo(2105, 358, 0.1)
    pyautogui.click()
    pyautogui.press('backspace')
    pyautogui.write(str(fp[1]), interval=0.2)
    
    auto_click(img_mode_path=r'mode_photo/register_enter.png')

def test(img_mode_path):
    '''
    检查当前屏幕截图中是否存在目标模板图像。
    Parameters
    ----------
    img_mode_path : String
        目标模版图路径

    Returns
    -------
    bool : True 表示匹配成功，False 表示匹配失败。
    '''
    # 截取当前屏幕内容并保存到指定路径
    screenshot_path = r'process_image/screenshot.png'
    pyautogui.screenshot().save(screenshot_path)
    
    # 载入截图和模板图像
    img = cv2.imread(screenshot_path)
    img_goal = cv2.imread(img_mode_path)
    
    if img is None or img_goal is None:
        print("无法加载截图或模板图像，请检查路径！")
        return False

    # 转为灰度图
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_goal_gray = cv2.cvtColor(img_goal, cv2.COLOR_BGR2GRAY)

    # 模板匹配
    result = cv2.matchTemplate(img_gray, img_goal_gray, cv2.TM_CCOEFF_NORMED)

    # 获取匹配度最高的值
    _, max_val, _, _ = cv2.minMaxLoc(result)

    # 匹配成功的阈值
    threshold = 0.8
    if max_val >= threshold:
        return True
    else:
        return False

def wait_result(img_path, interval=2, wait_time=20):
    '''
    等待程序响应。
    Parameters
    ----------
    img_mode_path : String
        目标模版图路径
    interval : int
        检查的时间间隔
    wait_time : int
        最长等待时间

    Returns
    -------
    bool : True 表示程序响应，False 超时。
    '''
    start_time = time.time()
    waiting = True
    while waiting:
        time.sleep(interval)
        if test(img_path):
            waiting = False
            return True

        elapsed_time = time.time() - start_time
        if elapsed_time > wait_time:
            waiting = False
            return False

def register_image(ip, fp):
    '''
    自动注册图像
    参数：
        ip: Initial Point 初始点坐标
        fp: Final Point 最终点坐标
    '''
    pyautogui.hotkey('Ctrl', 'r')
    # 输入起始点和最终点的坐标
    auto_click(img_mode_path=r'mode_photo\ipx.png')
    pyautogui.press('backspace')
    pyautogui.write(str(ip[0]), interval=0.2)
    
    auto_click(img_mode_path=r'mode_photo\ipy.png', duration=0.1)
    pyautogui.press('backspace')
    pyautogui.write(str(ip[1]), interval=0.2)
    
    auto_click(img_mode_path=r'mode_photo\fpx.png', duration=0.1)
    pyautogui.press('backspace')
    pyautogui.write(str(fp[0]), interval=0.2)

    auto_click(img_mode_path=r'mode_photo\fpy.png', duration=0.1)
    pyautogui.press('backspace')
    pyautogui.write(str(fp[1]), interval=0.2)
    
    auto_click(img_mode_path=r'mode_photo/register_enter.png')


