#include "Parameters.h"


Parameters::Parameters()
{
}


Parameters::~Parameters()
{
}


bool isDemo(shared_ptr<Parameter<int>> par)
{
	return par->key == "demo";
}
bool isCamera(shared_ptr<Parameter<bool>> par)
{
	return par->key == "c";
}

struct eqfloat{
	eqfloat(string compare_to) : compare_to_(compare_to) { }
	bool operator()(const shared_ptr<Parameter<float>> c1) const { return  c1->key == compare_to_; }
private:
	const string compare_to_;
};

void Parameters::clear()
{
	//for (auto &it : fparams) delete it;
	fparams.clear();

	vector<shared_ptr<Parameter<bool>>>::iterator cit = find_if(bparams.begin(), bparams.end(), isCamera);
	shared_ptr<Parameter<bool>> cparam(new Parameter<bool>((*cit)->name, (*cit)->value, (*cit)->key));
	//for (auto &it : bparams) delete it;
	bparams.clear();
	bparams.push_back(cparam);

	vector<shared_ptr<Parameter<int>>>::iterator dit = find_if(iparams.begin(), iparams.end(), isDemo);
	std::shared_ptr<Parameter<int>> dparam(new Parameter<int>((*dit)->name, (*dit)->value, (*dit)->key));
	//for (auto it : iparams) delete it;
	iparams.clear();
	iparams.push_back(dparam);
}


void Parameters::rem_float(const string key)
{
	vector<shared_ptr<Parameter<float>>>::iterator it = find_if(fparams.begin(), fparams.end(), eqfloat(key));
	
	if (it != fparams.end()) fparams.erase(it);
}

float Parameters::get_float(string name) const
{
	for (auto &it : fparams)
	{
		if (it->name == name) return it->value;
	}
	throw invalid_argument("invalid float arg");
}
bool Parameters::get_bool(string name) const
{
	for (auto it : bparams)
	{
		if (it->name == name) return it->value;
	}
	throw invalid_argument("invalid bool arg");
}
int Parameters::get_int(string name) const
{
	for (auto it : iparams)
	{
		if (it->name == name) return it->value;
	}
	throw invalid_argument("invalid int arg");
}


void Parameters::change(string key, string val)
{
	for (auto &it : bparams)
	{
		if ((*it).key == key)
		{
			if ((val == "true") || (val == "false"))
			{
				(*it).value = (val == "true");
			}
			return;
		}
	}
	for (auto &it : iparams)
	{
		if ((*it).key == key)
		{
			(*it).value = stoi(val);
			return;
		}
	}
	for (auto &it : fparams)
	{
		if ((*it).key == key)
		{
			(*it).value = stof(val);
			return;
		}
	}
}

std::ostream& operator<< (std::ostream& stream, const Parameter<bool>& param)
{
	stream << param.name << "(" << param.key << ")" << "=" << (param.value ? "true" : "false");
	return stream;
}