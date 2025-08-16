// src/pages/CompanyDetailPage.tsx
import { useEffect, useState, useMemo } from 'react';
import { useParams, Link } from 'react-router-dom';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button'; // For potential actions
import { ArrowLeft, MoreHorizontal, ArrowUpDown, ChevronLeft, ChevronRight } from 'lucide-react';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  // DropdownMenuSeparator, // Add if more actions
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useNavigate } from 'react-router-dom';


const API_BASE_URL = 'http://127.0.0.1:8000';
const USERS_PER_PAGE_COMPANY_DETAIL = 5; // Paginate users list on company page

// Re-using User and UserHealthScore interfaces (consider moving to a shared types.ts)
interface UserHealthScore { score: number; status: string; calculated_at: string; }
interface User { id: number; email: string; company_id: number; created_at: string; latest_health_score?: UserHealthScore | null; company_name?: string; } // company_name might be redundant here
interface Company { id: number; name: string; created_at: string; }

// Sorting for users list within company page
type UserSortKey = keyof Pick<User, 'id' | 'email' | 'created_at'> | 'health_score' | 'status';
interface UserSortConfig {
  key: UserSortKey;
  direction: 'ascending' | 'descending';
}


const CompanyDetailPage = () => {
  const { companyId } = useParams<{ companyId: string }>();
  const navigate = useNavigate();

  const [company, setCompany] = useState<Company | null>(null);
  const [usersInCompany, setUsersInCompany] = useState<User[]>([]); // All users fetched for this company
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // State for users table within company page
  const [userFilterTerm, setUserFilterTerm] = useState('');
  const [userSortConfig, setUserSortConfig] = useState<UserSortConfig | null>({ key: 'id', direction: 'ascending' });
  const [currentUsersPage, setCurrentUsersPage] = useState(1);


  useEffect(() => {
    if (!companyId) {
        setError("Company ID is missing.");
        setLoading(false);
        return;
    }

    const fetchCompanyData = async () => {
      setLoading(true);
      setError(null);
      try {
        // Fetch company details
        const companyRes = await fetch(`${API_BASE_URL}/companies/${companyId}`);
        if (!companyRes.ok) {
          if (companyRes.status === 404) throw new Error(`Company with ID ${companyId} not found.`);
          throw new Error(`Failed to fetch company details. Status: ${companyRes.status}`);
        }
        const companyData: Company = await companyRes.json();
        setCompany(companyData);

        // Fetch all users (then filter client-side for this company)
        // OR: If backend supports /companies/{companyId}/users, use that.
        // For now, let's assume we fetch all users and filter, similar to DashboardPage's company user count.
        // This is NOT ideal for many users but works for MVP.
        const allUsersRes = await fetch(`${API_BASE_URL}/users/?limit=500`); // Fetch a larger limit
        if (allUsersRes.ok) {
          const allUsersData: User[] = await allUsersRes.json();
          const companyUsersData = allUsersData.filter(u => u.company_id === parseInt(companyId));

          // Fetch health scores for these company users
          const usersWithHealth = await Promise.all(
            companyUsersData.map(async (user) => {
              try {
                const healthScoreRes = await fetch(`${API_BASE_URL}/users/${user.id}/latest_health_score`);
                const healthScoreData = healthScoreRes.ok ? await healthScoreRes.json() : null;
                return { ...user, latest_health_score: healthScoreData, company_name: companyData.name };
              } catch (e) {
                console.error(`Failed to fetch health score for user ${user.id} in company ${companyId}`, e);
                return { ...user, latest_health_score: null, company_name: companyData.name };
              }
            })
          );
          setUsersInCompany(usersWithHealth);

        } else {
          console.error(`Failed to fetch users list to filter for company ${companyId}. Status: ${allUsersRes.status}`);
          setUsersInCompany([]);
        }

      } catch (err) {
        if (err instanceof Error) setError(err.message);
        else setError('An unknown error occurred');
        console.error("Error in CompanyDetailPage fetchCompanyData:", err);
        setCompany(null); // Ensure company is null if its fetch fails
      } finally {
        setLoading(false);
      }
    };

    fetchCompanyData();
  }, [companyId]);

  // Memoized and sorted/filtered users for this company's table
  const processedCompanyUsers = useMemo(() => {
    let items = [...usersInCompany];
    if (userFilterTerm) {
      const lowerFilter = userFilterTerm.toLowerCase();
      items = items.filter(user => user.email.toLowerCase().includes(lowerFilter));
    }
    if (userSortConfig) {
      items.sort((a, b) => {
        let valA: any = ''; let valB: any = '';
        if (userSortConfig.key === 'health_score') {
          valA = a.latest_health_score?.score ?? -1;
          valB = b.latest_health_score?.score ?? -1;
        } else if (userSortConfig.key === 'status') {
          valA = a.latest_health_score?.status ?? '';
          valB = b.latest_health_score?.status ?? '';
        } else if (userSortConfig.key in a) {
          valA = (a as any)[userSortConfig.key];
          valB = (b as any)[userSortConfig.key];
        }
        valA = valA === null || valA === undefined ? '' : valA;
        valB = valB === null || valB === undefined ? '' : valB;
        if (typeof valA === 'string' && typeof valB === 'string') { valA = valA.toLowerCase(); valB = valB.toLowerCase(); }
        if (valA < valB) return userSortConfig.direction === 'ascending' ? -1 : 1;
        if (valA > valB) return userSortConfig.direction === 'ascending' ? 1 : -1;
        return 0;
      });
    }
    return items;
  }, [usersInCompany, userFilterTerm, userSortConfig]);

  const totalUsersSubPages = Math.ceil(processedCompanyUsers.length / USERS_PER_PAGE_COMPANY_DETAIL);
  const paginatedCompanyUsers = useMemo(() => {
    const start = (currentUsersPage - 1) * USERS_PER_PAGE_COMPANY_DETAIL;
    return processedCompanyUsers.slice(start, start + USERS_PER_PAGE_COMPANY_DETAIL);
  }, [processedCompanyUsers, currentUsersPage]);

  const requestUserSort = (key: UserSortKey) => {
    let direction: 'ascending' | 'descending' = 'ascending';
    if (userSortConfig?.key === key && userSortConfig.direction === 'ascending') {
      direction = 'descending';
    }
    setUserSortConfig({ key, direction });
    setCurrentUsersPage(1);
  };

  const getStatusColorClasses = (status?: string): string => { // Copied from DashboardPage
    switch (status) {
      case 'At Risk': return 'bg-red-300 hover:bg-red-400 text-red-800';
      case 'Power User': return 'bg-green-200 hover:bg-green-300 text-green-800';
      case 'Healthy': return 'bg-blue-300 hover:bg-blue-400 text-blue-800';
      case 'Neutral': return 'bg-gray-300 hover:bg-gray-400 text-gray-800';
      default: return 'bg-gray-100 hover:bg-gray-200 text-gray-800';
    }
  };

  if (loading) return <div className="flex justify-center items-center h-screen"><p>Loading company details...</p></div>;
  if (error) return <div className="text-red-500 p-4 bg-red-100 border border-red-300 rounded text-center">Error: {error}</div>;
  if (!company) return <div className="text-center p-8 text-muted-foreground">Company data could not be loaded or company not found.</div>;

  const SortableUserTableHead = ({ sortKey, children }: { sortKey: UserSortKey; children: React.ReactNode }) => (
    <TableHead onClick={() => requestUserSort(sortKey)} className="cursor-pointer hover:bg-muted/50">
      <div className="flex items-center">
        {children}
        {userSortConfig?.key === sortKey ? (
          <ArrowUpDown className={`ml-2 h-4 w-4 ${userSortConfig.direction === 'descending' ? 'opacity-100 rotate-180' : 'opacity-100'}`} />
        ) : (
          <ArrowUpDown className="ml-2 h-4 w-4 opacity-30" />
        )}
      </div>
    </TableHead>
  );

  return (
    <div className="space-y-6">
      <Link to="/" className="inline-flex items-center text-sm font-medium text-primary hover:underline mb-4">
        <ArrowLeft className="mr-2 h-4 w-4" />
        Back to Dashboard
      </Link>

      <Card>
        <CardHeader>
          <CardTitle className="text-2xl">{company.name}</CardTitle>
          <CardDescription>
            Company ID: {company.id} | Created: {new Date(company.created_at).toLocaleDateString()}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {/* Add more company-specific details here if available */}
          <p className="text-sm text-muted-foreground">Number of Users: {usersInCompany.length}</p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>Users in {company.name} ({processedCompanyUsers.length})</CardTitle>
            <Input 
              type="search" 
              placeholder="Filter users by email..." 
              value={userFilterTerm}
              onChange={(e) => {setUserFilterTerm(e.target.value); setCurrentUsersPage(1);}}
              className="max-w-xs"
            />
          </div>
        </CardHeader>
        <CardContent className="p-0">
          {paginatedCompanyUsers.length > 0 ? (
            <>
              <Table>
                <TableHeader>
                  <TableRow>
                    <SortableUserTableHead sortKey="id">User ID</SortableUserTableHead>
                    <SortableUserTableHead sortKey="email">Email</SortableUserTableHead>
                    <SortableUserTableHead sortKey="health_score">Health Score</SortableUserTableHead>
                    <SortableUserTableHead sortKey="status">Status</SortableUserTableHead>
                    <SortableUserTableHead sortKey="created_at">Joined</SortableUserTableHead>
                    <TableHead className="text-right">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {paginatedCompanyUsers.map(user => (
                    <TableRow key={user.id}>
                      <TableCell className="font-medium">{user.id}</TableCell>
                      <TableCell>{user.email}</TableCell>
                      <TableCell className="text-center">{user.latest_health_score?.score ?? 'N/A'}</TableCell>
                      <TableCell className="text-center">
                        {user.latest_health_score ? (
                          <Badge className={getStatusColorClasses(user.latest_health_score.status)}>
                            {user.latest_health_score.status}
                          </Badge>
                        ) : <Badge className={getStatusColorClasses(undefined)}>N/A</Badge>}
                      </TableCell>
                      <TableCell>{new Date(user.created_at).toLocaleDateString()}</TableCell>
                      <TableCell className="text-right">
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button variant="ghost" className="h-8 w-8 p-0">
                              <MoreHorizontal className="h-4 w-4" />
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end">
                            <DropdownMenuLabel>User Actions</DropdownMenuLabel>
                            <DropdownMenuItem onClick={() => navigate(`/users/${user.id}`)}>
                              View User Details
                            </DropdownMenuItem>
                            {/* Add more user actions if needed from company context */}
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
              {totalUsersSubPages > 1 && (
                <div className="flex items-center justify-end space-x-2 py-4 px-4 border-t">
                  <Button variant="outline" size="sm" onClick={() => setCurrentUsersPage(p => Math.max(1, p - 1))} disabled={currentUsersPage === 1}>
                    <ChevronLeft className="h-4 w-4 mr-1" /> Previous
                  </Button>
                  <span className="text-sm text-muted-foreground">Page {currentUsersPage} of {totalUsersSubPages}</span>
                  <Button variant="outline" size="sm" onClick={() => setCurrentUsersPage(p => Math.min(totalUsersSubPages, p + 1))} disabled={currentUsersPage === totalUsersSubPages}>
                    Next <ChevronRight className="h-4 w-4 ml-1" />
                  </Button>
                </div>
              )}
            </>
          ) : (
            <p className="p-4 text-sm text-muted-foreground text-center">
              {userFilterTerm ? "No users match your filter in this company." : "No users found in this company."}
            </p>
          )}
        </CardContent>
      </Card>
      {/* Could add sections for company-wide events or interventions if data model supports */}
    </div>
  );
};

export default CompanyDetailPage;