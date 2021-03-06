import pygame_menu
import pygame
import numpy

from keras import models

win = pygame.display.set_mode((500,500))
pixels = []



#region general ui
background = pygame_menu.panel(win,
    {
        "pos": (0,0),
        "size": (500,500),
        "color": (130,168,200),
    })

foreground = pygame_menu.panel(win,
    {
        "pos": (15,15),
        "size": (470,470),
        "color": (160,198,230),
        "parent": background,
        "outline": True,
        "outline_size": 1,
    })

def clearScreen():
    
    for row in pixels:
        for pixel in row:
            pixel.color = (0,0,0)

button_clear = pygame_menu.button(win,
    {
        "pos": (405,5),
        "size": (60,30),
        "color": (220,220,230),
        "parent": foreground,
        "color_hover": (210,210,220),
        "color_clicked": (200,200,210),
        "function": clearScreen,
    })

pygame_menu.text(win,
    {
        "text": "clear",
        "pos": (29,24),
        "size": 20,
        "parent": button_clear,
    })

def queryAi():

    #format pixels (panels) into 'mnist-like' image
    image_raw = []
    for y in pixels:
        row = []
        for x in y:
            row.append(sum(x.color)/3)
        image_raw.append(row)
    
    image = numpy.array(image_raw)
    image = numpy.reshape(image,(1,28,28,1))

    classifier = models.load_model('classifier.ai')
    output = list(classifier.predict(image)[0])
    digit_value = output.index(max(output))
    
    print(digit_value)
    
    


button_query = pygame_menu.button(win,
    {
        "pos": (405,40),
        "size": (60,30),
        "color": (220,220,230),
        "parent": foreground,
        "color_hover": (210,210,220),
        "color_clicked": (200,200,210),
        "function": queryAi,


    })

pygame_menu.text(win,
    {
        "text": "query",
        "pos": (29,24),
        "size": 19,
        "parent": button_query,
    })
#endregion

#region drawspace
drawspace_panel = pygame_menu.panel(win,
    {
        "pos":(70,150),
        "parent": background,
        "size":(280,280),
        "color": (255,255,255),
    })


#generate drawing panels
for y in range(28):

    row = []

    for x in range(28):
        
        row.append(pygame_menu.panel(win,
            {
                "pos":(x*10,y*10),
                "color":(0,0,0),
                "size":(10,10),
                "parent": drawspace_panel,
            }))

    pixels.append(row)





#endregion


run = True
while run:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    #checking if mouse clicked
    left, _, _ = pygame.mouse.get_pressed()
    if left:
        
        for row in pixels:
            for pixel in row:
                if pixel.checkHover():
                    pixel.color = (255,255,255)

    background.draw()

    pygame.display.update()