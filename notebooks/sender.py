# 发送方
from PIL import Image
import io

import sys, Ice
import Demo

with Ice.initialize(sys.argv) as communicator:
    # 发送
    base = communicator.stringToProxy("SimplePrinter:default -p 10000")
    printer = Demo.ImageTransferPrx.checkedCast(base)
    if not printer:
        raise RuntimeError("Invalid proxy")

    printer.printString("/data/home/suma/Downloads/plate_information/plate/00001_202501161427.jpg")

exit(0)

class ImageHandlerI(ImageProcessing.ImageHandler):
    def sendImage(self, imageData, plateStr, current):
        print(f"Received image data and plate number: {plateStr}")

        # 将字节数据转换为图像
        image = Image.open(io.BytesIO(imageData))
        image.show()  # 显示接收的图像

        # 图像处理示例：保存接收的图片
        image.save(f"{plateStr}_received.jpg")
        print(f"Image from plate {plateStr} saved successfully.")

if __name__ == "__main__":
    with Ice.initialize(sys.argv) as communicator:
        adapter = communicator.createObjectAdapterWithEndpoints("ImageHandlerAdapter", "default -p 10000")
        handler = ImageHandlerI()
        adapter.add(handler, communicator.stringToIdentity("ImageHandler"))
        adapter.activate()
        print("Server is running...")
        communicator.waitForShutdown()
