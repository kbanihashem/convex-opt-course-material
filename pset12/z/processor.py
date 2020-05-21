while True:
    text = input()
    if text == 'end':
        break
    if text[0] == 't':
        print(text)
        continue
    print(f"        [{','.join(text.split())}],")

