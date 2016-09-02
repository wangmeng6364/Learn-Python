# 实例：爬取python相关一千个网址

# 先导入有关的库，这一步很重要
import urllib2
from bs4 import BeautifulSoup   # 很好用的解析器
import re            # 很容易遗漏，即使漏掉程序也不报错，很隐蔽

# 程序的入口类
class SpiderMain(object):
    # 初始化对象
    def __init__(self):
        self.urls = UrlManger()
        self.downloader = HtmlDownloader()
        self.parser = HtmlParser()
        self.outputer = HtmlOutputer()
    
    # 最核心的方法
    def craw(self, root_url):
        count = 1  # 方便计数
        self.urls.add_new_url(root_url)
        while self.urls.has_new_url():
            try:
                # 取出url
                new_url = self.urls.get_new_url()
                print 'craw %d : %s' % (count, new_url)
                # 根据url下载网页内容
                html_cont = self.downloader.download(new_url)
                # 将网页内容解析为需要的数据
                new_urls, new_data = self.parser.parse(new_url, html_cont)
                # 将数据各自存放
                self.urls.add_new_urls(new_urls)
                self.outputer.collect_data(new_data)
                
                if count == 10:
                    break
                    
                count += 1
                
            except:
                print 'craw failed'
        
        # 将数据输出为html文件    
        self.outputer.output_html()
        

 
class UrlManger(object):

    # 初始化，两个set()，因为set()里的数据不会重复
    def __init__(self):
        self.new_urls = set()
        self.old_urls = set()
    
    # 添加一个url    
    def add_new_url(self, url):
        if url is None:
            return
        if url not in self.new_urls and url not in self.old_urls:
            self.new_urls.add(url)
    
    # 添加多个url，循环使用add_new_url，很巧妙  
    def add_new_urls(self, urls):
        if urls is None or len(urls) == 0:
            return 
        for url in urls:
            self.add_new_url(url)
    
    # 判断是否还有没有爬取的url，返回bool值
    def has_new_url(self):
        return len(self.new_urls) != 0
    
    # 获取一个url    
    def get_new_url(self):
        new_url = self.new_urls.pop() # pop用的极好，不仅取得了数据，而且将元素从set()中删除
        self.old_urls.add(new_url)   # 使用过的url要放在old_urls里，避免重复爬取
        return new_url

# 下载器        
class HtmlDownloader(object):
    
    def download(self, url):
        if url is None:
            return None
        
        # Python自带的urllib2下载模块        
        response = urllib2.urlopen(url)
        
        # 可以用getcode()方法判断是否下载成功
        if response.getcode() != 200:
            return None
            
        return response.read()

# Html解析器，核心，倒着看        
class HtmlParser(object):

    def get_new_urls(self, soup):
         
        # 解析出来的url也放在一个set()里
        new_urls = set()
        # 正则表达式是根据网页源代码分析出来的
        links = soup.find_all('a', href=re.compile(r'/view/\d+\.htm'))
        for link in links:
            new_url = link['href']
            # 解析出来的链接是个相对url，要合成成一个绝对url
            page_url = 'http://baike.baidu.com'
            new_full_url = page_url + new_url
            new_urls.add(new_full_url)
        return new_urls
    
    def get_new_data(self, page_url, soup):
    
        # 先构造一个字典
        res_data = {}
        
        # url
        res_data['url'] = page_url
        
        # <dd class="lemmaWgt-lemmaTitle-title"><h1>Python</h1>
        title_node = soup.find('dd', class_='lemmaWgt-lemmaTitle-title').find('h1')
        res_data['title'] = title_node.get_text()
        
        # <div class="lemma-summary" label-module="lemmaSummary">
        summary_node = soup.find('div', class_='lemma-summary')
        res_data['summary'] = summary_node.get_text()
        
        return res_data
        
    def parse(self, page_url, html_cont):
    
        if page_url is None or html_cont is None:
            return 
        
        # BeautifulSoup固定用法，先构造一个BeautifulSoup对象        
        soup = BeautifulSoup(html_cont, 'html.parser', from_encoding='utf-8')
        
        # 方法里面嵌套方法，值得学习
        new_urls = self.get_new_urls(soup)
        new_data = self.get_new_data(page_url, soup)
        return new_urls, new_data

# 输出器
class HtmlOutputer(object):
    
    def __init__(self):
        self.datas = []
        
    def collect_data(self, data):
        if data is None:
            return
        self.datas.append(data)
    
    # 输出为html文件    
    def output_html(self):
        # 以写的形式打开文件
        fout = open('D:/Learn/Code/python/pachong/output.html', 'w')
        
        # 这一句必须加上，否则中文字符会显示乱码
        fout.write('<meta charset="utf-8" />')
        fout.write('<html>')
        fout.write('<body>')
        fout.write('<table>')
        fout.write('<th>URL</th>')
        fout.write('<th>Title</th>')
        fout.write('<th>Summary</th>')
        
        for data in self.datas:
            fout.write('<tr>')
            fout.write('<td>%s</td>' % data['url'])
            # 注意中文字符的编码
            fout.write('<td>%s</td>' % data['title'].encode('utf-8'))
            fout.write('<td>%s</td>' % data['summary'].encode('utf-8'))
            fout.write('</tr>')
            
        fout.write('</table>')
        fout.write('</body>')
        fout.write('</html>')
        
        # 不要忘记关闭文件
        fout.close()

# 初始化程序
root_url = 'http://baike.baidu.com/view/21087.htm'
obj_spider = SpiderMain()
obj_spider.craw(root_url)
