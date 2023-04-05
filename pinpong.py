# 写一个python pingpong小游戏
import pygame

# 初始化pygame
pygame.init()

# 创建游戏窗口
screen = pygame.display.set_mode((800, 600))

# 创建球和挡板
ball = pygame.Rect(400, 300, 20, 20)
paddle = pygame.Rect(350, 550, 100, 10)

# 设置游戏背景颜色
bg_color = (255, 255, 255)

# 定义球的运动轨迹
ball_speed_x = 5
ball_speed_y = 5

# 定义挡板的运动轨迹
paddle_speed = 0

# 碰撞检测
if ball.colliderect(paddle):
    ball_speed_y = -ball_speed_y
# 定义游戏得分
score = 0

# 定义游戏结束条件
game_over = False
if ball.bottom > 600:
    game_over = True
# 游戏循环
while not game_over:
    # 处理事件
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_over = True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                paddle_speed = -5
            elif event.key == pygame.K_RIGHT:
                paddle_speed = 5
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                paddle_speed = 0

    # 更新游戏元素状态
    ball.x += ball_speed_x
    ball.y += ball_speed_y
    paddle.x += paddle_speed

    # 碰撞检测
    if ball.left < 0 or ball.right > 800:
        ball_speed_x = -ball_speed_x
    if ball.top < 0:
        ball_speed_y = -ball_speed_y
    if ball.bottom > 600:
        game_over = True
    if ball.colliderect(paddle):
        ball_speed_y = -ball_speed_y
        score += 1

    # 绘制游戏画面
    screen.fill(bg_color)
    pygame.draw.rect(screen, (255, 0, 0), ball)
    pygame.draw.rect(screen, (0, 0, 255), paddle)
    pygame.display.update()

# 退出pygame
pygame.quit()