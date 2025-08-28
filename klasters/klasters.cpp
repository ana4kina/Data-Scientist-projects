#include <iostream>
#include <vector>
#include <random>
#include <time.h>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <math.h>
#include <cstdlib>
#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <string>
#include <algorithm>

using namespace std;

unsigned int PRNG()
{
    static unsigned int seed = 2764;
    seed = (8259 * seed + 2303 * time(0));
    return seed % 32767;
}

double doubrand(double f1, double f2) {
    double f = (double)rand() / RAND_MAX;
    return (f1 + f * (f1 - f2));
}

double grand(int x) {  //Normal generation
    double r, v1, v2 = 0, faq;
    r = 6;
    while (r > 5) {
        srand(PRNG());
        v1 = (2 * rand() / (double)RAND_MAX - 1);
        v2 = (2 * rand() / (double)RAND_MAX - 1);
        r = v1 * v1 + v2 * v2;
    }
    faq = sqrt(abs(-2 * log(r) / r));
    return (v2 * faq);
}

class point {
public:
    point() {
        x = 0;
        y = 0;
        L = 0;
        int r = 255;
        int g = 255;
        int b = 255;
    }

    point(const point& other) {// Copy constructor             
        this -> x = other.x;             
        this -> y = other.y;             
        this -> L = other.L;         
    }

    point(double _x, double _y) {
        x = _x;
        y = _y;
        L = 0;
        int r = 255;
        int g = 255;
        int b = 255;
    }

    point(double _x, double _y, int _L) {
        x = _x;
        y = _y;
        L = _L;
        int r = 255;
        int g = 255;
        int b = 255;
    }

    void plusxy(double _x, double _y) {
        x += _x;
        y += _y;
    }
    void setx(double xx) {
        x = xx;
    }
    void sety(double yy) {
        y = yy;
    }
    void setr(int rr) {
        r = rr;
    }
    void setg(int gg) {
        g = gg;
    }
    void setb(int bb) {
        b = bb;
    }
    void setlabel(int _L) {
        L = _L;
    }
    double returnx() {
        return(x);
    }
    double returny() {
        return(y);
    }
    int returnlabel() {
        return (L);
    }
    bool operator== (const point& c1)             
    {
        return (x == c1.x &&
            y == c1.y);
    }

    /*
    void rotateof2d(float cx, float cy, float angle) {  //rotate around set point
        double pi = 3.1415926535;
        double rad = 0;
        rad = (angle * (pi / 180));

        float s = sin(rad);
        float c = cos(rad);

        // translate point back to origin:
        x -= cx;
        y -= cy;

        // rotate point
        float xnew = x * c - y * s;
        float ynew = x * s + y * c;

        // translate point back:
        x = xnew + cx;
        y = ynew + cy;
    }
    */

    /*void vivodfile() {
        //working file writer
        ofstream file;
        file.open("result.txt", std::ios::app);
        if (file.is_open())
        {
            file << "   X= " << setw(10) << x << "   Y= " << setw(10) << y << "   Label= " << setw(7) << L << endl;
            file.close();
        }
    }
    */

private:
    double x, y;
    int L;
    int r, g, b;
};

class cloud {
public:
    cloud() {
        n = 0;
        vect.resize(0);
        x = 0;
        y = 0;
        rx = 0;
        ry = 0;
        L = 0;
    }
   /* point ppoint(double x, double y, double rx, double ry) {
        srand(PRNG());

        int theta = rand() % 360;
        double rr = rand() % (int)rx;
        double xx = x/10 + grand((int)rr) * cos(theta) * 5;
        double yy = y/10 + grand((int)rr) * sin(theta) * (ry / rx) * 5;
        xx *= 10000;
        xx = (int)xx;
        xx /= 1000;
        yy *= 10000;
        yy = (int)yy;
        yy /= 1000;
        return (point(xx, yy, L));
    }
    */

    void addpoint(double x, double y, double rx, double ry) {                     //dobavlenie tochki v vector
        srand(PRNG());

        int theta = rand() % 360;
        double rr = rand() % (int)rx;
        double xx = x + grand((int)rr) * cos(theta) * 5 * 10;
        double yy = y + grand((int)rr) * sin(theta) * (ry / rx) * 5 * 10;
        /*
        xx = (int)xx;
        yy = (int)yy;
        */

        vect.push_back(point(xx, yy, 0));
    }

    cloud(int _n, double _x = 0, double _y = 0, double _rx = 0, double _ry = 0, int _L = 0) {
        n = _n;
        vect.resize(0);
        x = _x;
        y = _y;
        rx = _rx;
        ry = _ry;
        L = _L;
        for (int i = 0; i < _n; i++) {
            addpoint(x, y, rx, ry);
        }
    }

    void plusxy(double _x, double _y) {             //f-ya izmeneniya znachenia x i y na 1
        x += _x;
        y += _y;
        for (int i = 0; i < vect.size(); i++) {
            vect[i].plusxy(_x, _y);
        }

    }

    void rotateaboutitself(double phi) {           //rotarion group relative (0;0)    //насколько я поняла, вращение вектора точек вокруг себя
        double newx, newy; 
        vector <point> newvect;
        newvect.reserve(vect.size());
        for (int i = 0; i < vect.size(); i++) {   //изменение координат точек в векторе и их поворот по формуле ниже
            newx = vect[i].returnx();
            newy = vect[i].returny();
            newvect.push_back(point(
                newx * cos(phi) - newy * sin(phi),
                newx * sin(phi) + newy * cos(phi)));
        }
        vect.clear();
        for (int i = 0; i < newvect.size(); i++) {
            vect.push_back(point(newvect[i]));
        }
    }


    void rotate00(double phi) {//rotarion group relative to group center // перенос центра
        double midx = 0, midy = 0, newx, newy;
        vector <point> newvect;
        newvect.reserve(vect.size());
        for (int i = 0; i < vect.size(); i++) {
            midx += vect[i].returnx();
            midy += vect[i].returny();
        }
        midx /= vect.size();
        midy /= vect.size();                      // всё до этого - поиск координат центра
        for (int i = 0; i < vect.size(); i++) {          //перенос
            newx = vect[i].returnx() - midx;
            newy = vect[i].returny() - midy;
            newvect.push_back(point(                               //поворот изменённого центра
                (newx * cos(phi) - newy * sin(phi)) + midx,
                (newx * sin(phi) + newy * cos(phi)) + midy));
        }
        vect.clear();                                  //очистка старого вектора
        for (int i = 0; i < newvect.size(); i++) {     //заполнение получившимися значениями
            vect.push_back(point(newvect[i]));
        }
    }

   /* void vivodharacter() {
        cout << "   Number of points: " << n << endl;;
        cout << "   Center (x,y): " << setw(5) << x << "  " << setw(5) << y << endl;
        cout << "   Dispersion by x and by y: " << setw(5) << rx << "  " << setw(5) << ry << endl;
        cout << "   Label: " << L << endl;
        cout << endl;
    }
    */
    /*
    void vivodfile() {
        ofstream file;
        file.open("result.txt", std::ios::app);
        if (file.is_open())
        {
            file << "cloud № " << L << endl;
            file.close();
        }
        for (int i = 0; i < vect.size(); i++) {
            vect[i].vivodfile();
        }
        file.open("result.txt", std::ios::app);
        if (file.is_open())
        {
            file << endl;
            file.close();
        }
    }
    */

    vector<point> ReturnVector() {
        return vect;
    }

    ~cloud() {                                     //деструктор
        vect.erase(vect.begin(), vect.end());
    }
    

private:
    int n;               //kol-vo tochek                       // кол-во точек
    vector <point>vect;  //vector of points                    // вектор из точек
    double x, y, rx, ry; //middle point (x, y), +- for x and y // центр(х, у), разброс по х и у
    int L;               //cloud's lable                       // метка облака
};


class Pole {
public:
    vector <point> P;

    Pole() {
        P.resize(0);
    }

   void addcloud(cloud obl) {                          //добавление точек в облако
       vector<point> vectt = obl.ReturnVector();       
       for (int i = 0; i < vectt.size(); i++) {
            P.push_back(point(vectt[i]));
       }
   }

    void vivodfile() {                       //с файловыми функциями я не ебусь, но тут запись координат точек облака в файл
        ofstream file("data.dat");
        for (int i = 0; i < P.size(); i++) {
            file << to_string(P[i].returnx()) << ' ' << to_string(P[i].returny()) << ' ' << "0 0 0" << endl;
        }
        file.close();
    }

    ~Pole() {                               //деструктор
        P.erase(P.begin(), P.end());
    }
};

class kkord {  //neponyatno zachem
public:
    double x, y;
    kkord(double xx = 0, double yy = 0) {
        x = xx;
        y = yy;
    }
};

class cluster {

public:

    void remathcentr() {                        //центр
        int count = 0;
        double xx = 0; 
        double yy = 0;
        for (int i = 0; i < vect.size(); i++) {
            xx += vect[i].returnx();
            yy += vect[i].returnx();
            count++;
        }
        x = xx / count;
        y = yy / count;
    }

    cluster() {
        n = 0;
        vect.resize(0);
        x = 0;
        y = 0;
        L = 0;
        r = rand() % 255;
        g = rand() % 255;
        b = rand() % 255;
    }

    void addpoint(point A) {              //добавление точки
        vect.push_back(point(A));
        remathcentr();
    }

    void pop_back() {                   //удаление последнего элемента вектора
        vect.pop_back();
    }

    int returnr() {
        return r;
    }

    int returng() {
        return g;
    }

    int returnb() {
        return b;
    }

    double returnx() {
        return x;
    }

    double returny() {
        return y;
    }

    vector<point> ReturnVector() {
        return vect;
    }

    ~cluster() {                                 //деструктор
        vect.erase(vect.begin(), vect.end());
    }


private:
    int n;               //kol-vo tochek
    vector <point>vect;  //vector of points
    double x, y;         //middle point (x, y)
    int L;               //cloud's lable
    int r, g, b;
};


class hierarchymethod {
public:
    vector <point> P;
    int count = 0;
    vector <cluster> cls;

    hierarchymethod(const vector<point> &PP,int n = 2) {
        P = PP;
        count = n;
        cls.resize(0);
        sortit(P);
    }

    double dist(point p1, point p2) {
        double distx = p2.returnx() - p1.returnx();
        double disty = p2.returny() - p1.returny();
        return(sqrt(distx*distx+disty*disty));
    }
    double dist(point p1, cluster p2) {
        double distx = p2.returnx() - p1.returnx();
        double disty = p2.returny() - p1.returny();
        return(sqrt(distx * distx + disty * disty));
    }
    double dist(cluster p1, cluster p2) {
        double distx = p2.returnx() - p1.returnx();
        double disty = p2.returny() - p1.returny();
        return(sqrt(distx * distx + disty * disty));
    }
    
    void VFile() {
        ofstream file("data.dat");
        for (int i = 0; i < cls.size(); i++) {
            for (int j = 0; j < cls[i].ReturnVector().size(); j++) {
                string s = to_string(cls[i].ReturnVector()[j].returnx()) + " " + to_string(cls[i].ReturnVector()[j].returny()) + " " + to_string(cls[i].returnr()) + " " + to_string(cls[i].returng()) + " " + to_string(cls[i].returnb());
                file << s << endl;
            }
        }
        file.close();
    }

    void sortit(const vector<point> &PP) {
        cls.resize(0);
        vector<point> pnt = PP;
        string where1 = "point", where2 = "point";
        int obj1 = 0, obj2 = 1;
        double mindist = dist(pnt[0], pnt[1]);
        if (pnt.size() > 1) {
            while (pnt.size() != 0 || cls.size() > count) {
                cout << "  " << mindist << endl << pnt.size() << " " << cls.size();
                mindist = 9999999999.99;

                for (int i = 0; i < pnt.size(); i++) {

                    for (int j = 0; j < pnt.size(); j++) {
                        if ((dist(pnt[i], pnt[j]) < mindist) && (i != j)) {
                            mindist = dist(pnt[i], pnt[j]);
                            where1 = "point";
                            where2 = "point";
                            obj1 = i;
                            obj2 = j;
                        }
                    }
                    for (int j = 0; j < cls.size(); j++) {
                        if (dist(pnt[i], cls[j]) < mindist) {
                            mindist = dist(pnt[i], cls[j]);
                            where1 = "point";
                            where2 = "cluster";
                            obj1 = i;
                            obj2 = j;
                        }
                    }
                }
                for (int i = 0; i < cls.size(); i++) {
                    for (int j = 0; j < cls.size(); j++) {
                        if ((dist(cls[i], cls[j]) < mindist) && (i != j)) {
                            mindist = dist(cls[i], cls[j]);
                            where1 = "cluster";
                            where2 = "cluster";
                            obj1 = i;
                            obj2 = j;
                        }
                    }
                }

                if (where1 == "cluster" && where2 == "cluster" && ((pnt.size()+cls.size()) > count) ) {
                    if (cls[obj1].ReturnVector().size() > cls[obj2].ReturnVector().size()) {
                        int z = obj1;
                        obj1 = obj2;
                        obj2 = z;
                    }
                    for (int i = 0; i < cls[obj1].ReturnVector().size(); i++) {
                        cls[obj2].addpoint(cls[obj1].ReturnVector()[i]);
                    }
                    cls.erase(cls.begin() + obj1);
                }
                else if (where1 == "point" && where2 == "cluster" && (pnt.size() + cls.size()) > count) {
                    cls[obj2].addpoint(pnt[obj1]);
                    pnt.erase(pnt.begin() + obj1);
                }
                else if (where1 == "point" && where2 == "point" && (pnt.size() + cls.size()) > count) {
                    cluster z;
                    z.addpoint(pnt[obj1]);
                    z.addpoint(pnt[obj2]);
                    cls.push_back(z);
                    if (obj1 < obj2) {
                        pnt.erase(pnt.begin() + obj2);
                        pnt.erase(pnt.begin() + obj1);
                    }
                    else {
                        pnt.erase(pnt.begin() + obj1);
                        pnt.erase(pnt.begin() + obj2);
                    }
                }
                else {
                    for (int i = pnt.size() - 1; i >= 0; i--) {
                        cluster z;
                        z.addpoint(pnt[i]);
                        cls.push_back(z);
                        pnt.erase(pnt.begin() + i);
                    }
                }
            }
        }
        else {
            cout << "You have less then 2 dots " << endl;
        }
        cout << pnt.size() << " " << cls.size() << endl;
    }

    ~hierarchymethod() {
        P.erase(P.begin(), P.end());
        cls.erase(cls.begin(), cls.end());
    }
};

class forelmethod {
public:
    vector <point> P;
    double radius = 0;
    vector <cluster> cls;
    const int a = 0.95;

    forelmethod(const vector<point>& PP, double r = 2) {
        P = PP;
        radius = r;
        cls.resize(0);
        sortit(P);
    }

    double dist(point p1, point p2) {
        double distx = p2.returnx() - p1.returnx();
        double disty = p2.returny() - p1.returny();
        return(sqrt(distx * distx + disty * disty));
    }

    void VFile() {
        ofstream file("data.dat");
        for (int i = 0; i < cls.size(); i++) {
            for (int j = 0; j < cls[i].ReturnVector().size(); j++) {
                string s = to_string(cls[i].ReturnVector()[j].returnx()) + " " + to_string(cls[i].ReturnVector()[j].returny()) + " " + to_string(cls[i].returnr()) + " " + to_string(cls[i].returng()) + " " + to_string(cls[i].returnb());
                file << s << endl;
            }
        }
        file.close();
    }

    void sortit(const vector<point>& PP)  {
        cls.resize(0);
        vector<point> pnt = PP;
        string where1 = "point", where2 = "point";
        int obj1 = 0, obj2 = 1;

        while (pnt.size() > 1) {  //poka ne clasterizovany vse elementy
            point p1 = pnt[0];  //beru _sluchainyi ob'ect
            cluster probclust;
            vector<int> vectornaudalenie;
            for (int i = 1; i < pnt.size(); i++) {
                if (dist(p1, pnt[i]) <= radius) { //dobavliau suda vse tochki v zadanom radiuse
                    probclust.addpoint(pnt[i]);
                }
            }
            probclust.remathcentr(); //schitaiu centr mass
            point curentcentr(probclust.returnx() - remainder(probclust.returnx(), 0.001), probclust.returny() - remainder(probclust.returny(), 0.001)); //point v centre mass potencial'nogo clastera
            for (int i = probclust.ReturnVector().size(); i > 0; i--) {
                probclust.pop_back();
            }
            point curentcentr1;
            while (curentcentr.returnx() != curentcentr1.returnx() && curentcentr.returny() != curentcentr1.returny()) {
                //cout << "x= " << curentcentr.returnx() << "   " << curentcentr1.returnx() << endl;
                //cout << "y= " << curentcentr.returny() << "   " << curentcentr1.returny() << endl;
                curentcentr = curentcentr1;
                vectornaudalenie.resize(0);
                for (int i = 0; i < pnt.size(); i++) {
                    if (dist(p1, pnt[i]) <= radius) { //dobavliau suda vse tochki v zadanom radiuse
                        probclust.addpoint(pnt[i]);
                        vectornaudalenie.push_back(i);
                    }
                }
                sort(vectornaudalenie.begin(), vectornaudalenie.end());

                curentcentr1.setx(probclust.returnx() - remainder(probclust.returnx(),0.001));
                curentcentr1.sety(probclust.returny() - remainder(probclust.returny(), 0.001)); //point v centre mass potencial'nogo clastera
            }
            cluster gotovyiclust;
            //cout << pnt.size() << " 1 " << cls.size() << endl;
            for (int i = vectornaudalenie.size() - 1; i >= 0; i--) {  //udalenie vibranyh ++ zapis' ih v claster
                gotovyiclust.addpoint(pnt[vectornaudalenie[i]]);
                if (vectornaudalenie[i] != pnt.size()) {
                    pnt.erase(pnt.begin() + vectornaudalenie[i]);
                }
                else {
                    pnt.pop_back();
                }
            }
            //cout << pnt.size() << " 2 " << cls.size() <<  endl;
            cls.push_back(gotovyiclust);
            for (int i = gotovyiclust.ReturnVector().size(); i > 0; i--) {
                gotovyiclust.pop_back();
            }
        }

    }

    ~forelmethod() {
        P.erase(P.begin(), P.end());
        cls.erase(cls.begin(), cls.end());
    }
};

class dbscanmethod {
public:
    vector <point> P;
    double radius = 0;
    vector <cluster> cls;
    int count;

    dbscanmethod(const vector<point>& PP, int cnt = 10, double r = 2) {
        P = PP;
        radius = r;
        count = cnt;
        cls.resize(0);
        sortit(P);
    }

    double dist(point p1, point p2) {
        double distx = p2.returnx() - p1.returnx();
        double disty = p2.returny() - p1.returny();
        return(sqrt(distx * distx + disty * disty));
    }

    void VFile() {
        ofstream file("data.dat");
        for (int i = 0; i < cls.size(); i++) {
            for (int j = 0; j < cls[i].ReturnVector().size(); j++) {
                string s = to_string(cls[i].ReturnVector()[j].returnx()) + " " + to_string(cls[i].ReturnVector()[j].returny()) + " " + to_string(cls[i].returnr()) + " " + to_string(cls[i].returng()) + " " + to_string(cls[i].returnb());
                file << s << endl;
            }
        }
        file.close();
    }

    //L == 0 - ne prosmotrennye /undefined
    //L == 1 - pometka shum
    //L == 2 - pometka kray
    //L == 3 - started dot
    bool ifend(vector <point> PP) {
        bool z = true;
        for (int i = 0; i < PP.size(); i++) {
            if (PP[i].returnlabel() == 0) {
                z = false;
            }
        }
        return(z);
    }

    void sortit(const vector<point>& PP) {
        cls.resize(0);
        vector<point> pnt = PP;

        while (ifend(pnt) != true) {

        }
       

    }

    ~dbscanmethod() {
        P.erase(P.begin(), P.end());
        cls.erase(cls.begin(), cls.end());
    }
};




class Finder {
public:
    vector<point> Vvect;
    Finder() {

    }
    void addpoint(point P) {
        Vvect.push_back(P);
    }
    void addpoints(vector<point> P) {
        for (int i = 0; i < P.size(); i++) {
            Vvect.push_back(P[i]);
        }
    }

    vector<point> returnvector() {
        return Vvect;
    }


    void forelmethod1(int r = 1) {
        forelmethod met(Vvect, r);
        met.VFile();
        system("C:/gnuplot/bin/gnuplot.exe ./plot.plt"); //a simple example function
    }

    void hierarchymethod1(int n = 2) { //vector of dots and count of clasters
        hierarchymethod met(Vvect, n);
        met.VFile();
        system("C:/gnuplot/bin/gnuplot.exe ./plot.plt"); //a simple example function
    }

    vector<point> sptrmethod(vector<point> Vect, int n) {
        return Vvect;
    }

    ~Finder() {
        Vvect.erase(Vvect.begin(), Vvect.end());
    }
};

class interface {
public:
    Finder finder;
    interface() {
        ofstream file;  //obnylenie fila
        file.open("result.txt");
        if (file.is_open())
        {
            file << "";
            file.close();
        }

        int whil = 1;
        while (whil != 0) {
            cout << " Set here your choice: " << endl;
            cout << " 1 - forel sorting method " << endl;
            cout << " 2 - hierarchy sorting method " << endl;
            cout << " 3 - sorting method №o 3 " << endl;
            cout << " 4 - create new cloud of dots " << endl;
            cout << " 5 - show all clouds of dots " << endl;
            cout << " 6 - add cloud to your field (cloud will be deleted) " << endl;
            cout << " 7 - show numbers of points in your field " << endl;

            cout << endl;
            cout << " 0 - stop this programm " << endl;
            cout << "Your input: ";
            cin >> whil;
            if (whil == 1) {
                cout << "Radius for clusteing: ";
                int r = 0;
                
                cin >> r;
                if (r == 0) {
                    while (r == 0) {
                        cout << "Radius == 0" << endl;
                        cout << "Try new radius: ";
                        cin >> r;
                    }
                }
                finder.forelmethod1(abs(r));

            }
            else if (whil == 2) {
                cout << "Count of clusters: ";
                int n = 0;
                cin >> n;
                finder.hierarchymethod1(abs(n));
            }
            else if (whil == 3) {
                //start method 3
            }
            /* else if (whil == -1) {
                ofstream file("data.dat");
                for (int i = 0; i < vectorofpoints.size(); i++) {
                    string s = to_string(vectorofpoints[i].returnx()) + " " + to_string(vectorofpoints[i].returny()) + " 0 0 0";
                    file << s << endl;
                }
                file.close();

                system("C:/gnuplot/bin/gnuplot.exe ./plot.plt"); //a simple example function

            }
            */
            else if (whil == 4) {
                int _n, _L; double _x, _y, _rx, _ry;
                cout << " Set here (int)numbers of points in cloud, (double)x,y of center, (double)rx,ry of dispersion by x and by y: " << endl;
                cout << "Set 0 for (int)numbers of points in cloud to close it " << endl;
                cout << "Set (int)numbers of points in cloud: ";
                cin >> _n;
                _n = abs(_n);
                if (_n != 0) {
                    cout << endl << "Set (double)x of center: ";
                    cin >> _x;
                    cout << endl << "Set (double)y of center: ";
                    cin >> _y;
                    cout << endl << "Set (double)rx of dispersion by x: ";
                    cin >> _rx;
                    cout << endl << "Set (double)ry of dispersion by y: ";
                    cin >> _ry;
                    cout << endl;

                    cloud z(_n, _x, _y, abs(_rx), abs(_ry));
                    finder.addpoints(z.ReturnVector());
                    cout << "   Cloud had been created" << endl;
                    cout << endl;
                }
            }
            else {
                     cout << "   No clouds created yet" << endl;
            }
        }
    }
    ~interface() {
    }
};

int main() {
    interface();
    return 0;
}