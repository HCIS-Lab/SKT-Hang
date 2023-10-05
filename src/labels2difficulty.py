import os, glob

if __name__=="__main__":

    input_file = 'labels_5.txt'
    output_file = 'labels_5c_difficulty.txt'

    assert os.path.exists(input_file), f'{input_file} not exists'

    f_input = open(input_file, 'r')
    input_lines = f_input.readlines()

    f_output = open(output_file, 'w')

    cls_dict = {}
    flag = False
    for cat_line in input_lines:
        if '==============================================================================================' in cat_line:
            flag = not flag
            continue
        
        if flag:
            line_strip = cat_line.strip()
            hook_name = line_strip.split('=>')[0].strip()
            val = line_strip.split('=>')[1].strip()
            new_line = '{}{},{}\n'.format(hook_name[:-len(hook_name.split('_')[-1])], val ,hook_name.split('_')[-1])
            f_output.write(new_line)
    
    f_output.close()