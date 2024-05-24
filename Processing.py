from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split

class CleaningData:
    def __init__(self):
        self.determined_labels = {'shop': 0, 'item': 1, 'total': 2, 'date_time': 3, 'other': 4}  

    def get_data(self, annotation_file, ext_word, split_rate=0.2):
        dt_set = []

        with open(annotation_file, 'r') as f:
            data = f.read()
        Bs_data = BeautifulSoup(data, "xml")
        infors = Bs_data.find_all("image")

        for i, idx in enumerate(infors):
            k = []
            z = []
            org_dt = []
            name_file = idx.get('name')
            result = [inner_list[0] for inner_list in ext_word[i][1]]
            boxes = idx.find_all('box')

            for box in boxes:
                attribute_element = box.find('attribute', {'name': 'text'})
                text_content = attribute_element.text if attribute_element else None
                label = box.get('label')
                k.append((text_content, label))
                z.append([text_content, label])
                org_dt = z.copy()
            for x in result:
                match_found = False
                for i, j in k:
                    if x in i:
                        z.append([x, j])
                        match_found = True
                        break
                if not match_found:
                    z.append([x, 'other'])
            dt_set.append([name_file, z + org_dt])
        train_dt, val_dt = train_test_split(dt_set, test_size=split_rate)

        return train_dt, val_dt

    def get_NLP_data(self, ds):
        X = []
        y = []
        for index in ds:
            for inf in index[1]:
                text = inf[0]
                label = self.determined_labels[inf[1]]
                X.append(text)
                y.append(label)
        return X, y
