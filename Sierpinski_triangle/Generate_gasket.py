from PIL import Image

def build(triangle, depth):
    if(depth <= 0):
        triangle.save("Triangle.png")
        return

    # Prepare canvas:
    new = Image.new("L", (triangle.height*2, triangle.height*2), 255)
    # Upper triangle:
    new.paste(triangle, (new.width//4 , 0))
    # Bottom-left:
    new.paste(triangle, (0, new.height//2))
    # Bottom-right:
    new.paste(triangle, (new.width//2 , new.height//2))

    build(new, depth-1)

tr = Image.open("single1.png")
tr = tr.resize((8,8))
tr.save("op_unit.png")
build(tr, 12)