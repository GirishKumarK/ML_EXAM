file = open('data.txt', 'r')
lines = file.read().split('\n')
new_lines = []
for line in lines:
    words = line.split('\t')
    for word in words:
        if (word == ''):
            words.remove(word)
    new_line = []
    for w in range(len(words)):
        new_line.append(words[w])
        new_line.append(',')
    new_lines.append(new_line[0 : -1])
file.close()

file = open('new_data.csv', 'w')
for nl in new_lines:
    nline = ''.join(nl)
    file.write(nline + '\n')
file.close()
