# -*- coding: utf-8 -*-

from variance import total_sale

# run the functions
while True:
    temperature = raw_input("\n예상온도는 몇 도입니까?\n(종료: q)\n")
    if temperature == "q":
        break
    try:
        temperature = int(temperature)
        if -50 <= temperature >= 60:
            print("적정 온도를 기입해주세요.")
        else:
            total_sale(temperature)
    except ValueError:
        print("숫자를 입력해주세요.\n")

