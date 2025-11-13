import regex

# adapted from kaggle.com/code/slawekbiel/resnet18-0-772-public-lb/notebook
measurements = {
    'weight': [('mg',1), ('g', 1000), ('gr', 1000), ('gram', 1000), ('kg', 1000000)],
    'length': [('mm',1), ('cm', 10), ('m',1000), ('meter', 1000)],
    'pieces': [ ('pc',1)],
    'memory': [('gb', 1)],
    'volume': [('ml', 1), ('l', 1000), ('liter',1000)]
}

def to_num(x, mult=1):
    x = x.replace(',','.')
    return int(float(x)*mult)

def extract_unit(tit, m):
    pat = rf'\W(\d+(?:[\,\.]\d+)?) ?{m}s?\W'
    matches = regex.findall(pat, tit, overlapped=True)
    return set(matches)

def extract(tit):
    res =dict()
    tit = ' '+tit.lower()+' '
    for cat, units in measurements.items():
        cat_values=set()
        for unit_name, mult in units:
            values = extract_unit(tit, unit_name)
            values = {to_num(v, mult) for v in values}
            cat_values = cat_values.union(values)
        if cat_values:
            res[cat] = cat_values
    return res

def match_measures(m1, m2):
    k1,k2 = set(m1.keys()), set(m2.keys())
    common = k1.intersection(k2)
    if not common:
        return True
    for key in common:
        s1,s2 = m1[key], m2[key]
        if s1.intersection(s2):
            return True
    return False

def filter_match_titles(matches : list, title_list): # filter matches override
    for i in range(len(matches)):
        item_title = extract(title_list)
        l=[]
        for match in matches[i]:
            if match == i:
                l.append(i)
                continue
            match_title = extract(title_list)
            if (match_measures(item_title, match_title)):
                l.append(match)
        matches[i] = l
    return matches