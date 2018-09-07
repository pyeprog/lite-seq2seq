conversation_file = 'movie_conversations.txt'
line_file = 'movie_lines.txt'

with open(conversation_file, 'r', encoding='utf-8', errors='ignore') as fp:
    raw_conversation = fp.readlines()

with open(line_file, 'r', encoding='utf-8', errors='ignore') as fp:
    raw_lines = fp.readlines()

line_dict = {}
conv_list = []

n_lines = len(raw_lines)
for i, line in enumerate(raw_lines):
    print('{}/{} Line processing ...'.format(i+1, n_lines))
    line_info = line.split(' +++$+++ ')
    line_dict[line_info[0]] = line_info[-1]
        

n_conversation = len(raw_conversation)
for i, conv in enumerate(raw_conversation):
    print('{}/{} conversation processing ...'.format(i+1, n_conversation))
    conv_list.append(eval(conv.split(' +++$+++ ')[-1])) 

# print(conv_list[:10])
# print(line_dict['L204'])

question_list = []
answer_list = []

n_conv_list = len(conv_list)
for c, conv in enumerate(conv_list):
    print('{}/{} Parsing questions and answers ...'.format(c+1, n_conv_list))
    for i in range(len(conv)-1):
        question_list.append(line_dict[conv[i]])
        answer_list.append(line_dict[conv[i+1]])

questions = ''.join(question_list).lower()
answers = ''.join(answer_list).lower()


with open('enc', 'w') as fp:
    fp.write(questions)

with open('dec', 'w') as fp:
    fp.write(answers)

print('All Done!')
